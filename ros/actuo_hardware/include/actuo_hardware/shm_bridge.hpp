// Header-only shared-memory seqlock, the C++ mirror of actuo's
// actuo/drivers/shm.py. Layout and constants MUST match that file; its
// docstring is the description of the format (layout version 2, magic 'ACT2').
//
// The Python bus servicer CREATES the segment and is the only side that ever
// zeroes it. Every ros2_control hardware component ATTACHES and writes only
// its own command slots, so one module can be deactivated and re-activated
// while the others keep running. Only flat float64 vectors cross this
// boundary: no protocol knowledge, so nothing to keep in step with Python
// except this layout.
#ifndef ACTUO_HARDWARE__SHM_BRIDGE_HPP_
#define ACTUO_HARDWARE__SHM_BRIDGE_HPP_

#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace actuo_hardware
{

class ShmBridge
{
public:
  // The bytes 'ACT2' read as a little-endian uint32 (shm.MAGIC).
  static constexpr uint32_t kMagic = 0x32544341;
  static constexpr size_t kHeaderSize = 64;
  static constexpr size_t kCmdSeqOff = 16;
  static constexpr size_t kStateSeqOff = 24;
  static constexpr size_t kEpochOff = 32;
  static constexpr uint32_t kCmdStride = 3;    // [position, velocity, effort]
  static constexpr uint32_t kStateStride = 3;  // [position, velocity, present]
  static constexpr size_t kStatePresent = 2;

  // File-name prefix: $ACTUO_SHM_PREFIX if set, else "actuo_" -- the same
  // rule as shm.shm_prefix(), so both sides compute the same path.
  static std::string prefix()
  {
    const char * env = std::getenv("ACTUO_SHM_PREFIX");
    return env != nullptr ? std::string(env) : std::string("actuo_");
  }

  static std::string default_path(const std::string & name)
  {
    return "/tmp/" + prefix() + name + ".shm";
  }

  static std::string lock_path(const std::string & name)
  {
    return "/tmp/" + prefix() + name + ".lock";
  }

  // True while a bus servicer holds this segment's lock. The servicer takes
  // an exclusive flock on the lock file for its whole life and the kernel
  // drops it however the process dies, so this is the authority on "is the
  // segment's state current". A file left by an earlier run is not.
  static bool servicer_alive(const std::string & name)
  {
    const int fd = ::open(lock_path(name).c_str(), O_RDONLY);
    if (fd < 0) { return false; }
    const bool held = ::flock(fd, LOCK_SH | LOCK_NB) != 0;
    if (!held) { ::flock(fd, LOCK_UN); }
    ::close(fd);
    return held;
  }

  // Attach to a segment the servicer created. Throws if it is absent, has
  // another magic, or is not fully written.
  explicit ShmBridge(const std::string & name, const std::string & path = "")
  : name_(name)
  {
    path_ = path.empty() ? default_path(name) : path;
    fd_ = ::open(path_.c_str(), O_RDWR);
    if (fd_ < 0) {
      throw std::runtime_error("actuo shm: no segment at " + path_);
    }
    struct stat st;
    if (::fstat(fd_, &st) != 0 || static_cast<size_t>(st.st_size) < kHeaderSize) {
      close_fd();
      throw std::runtime_error("actuo shm: segment shorter than its header: " + path_);
    }
    map(kHeaderSize);
    const uint32_t magic = u32(0);
    n_ = u32(4);
    const uint32_t cs = u32(8);
    const uint32_t ss = u32(12);
    unmap();
    if (magic != kMagic) {
      close_fd();
      throw std::runtime_error("actuo shm: wrong magic, not an actuo segment (or not yet written): " + path_);
    }
    if (cs != kCmdStride || ss != kStateStride) {
      close_fd();
      throw std::runtime_error("actuo shm: unexpected strides in " + path_);
    }
    size_ = kHeaderSize + static_cast<size_t>(n_) * (kCmdStride + kStateStride) * 8;
    if (static_cast<size_t>(st.st_size) < size_) {
      close_fd();
      throw std::runtime_error("actuo shm: segment shorter than its layout: " + path_);
    }
    map(size_);
    cmd_off_ = kHeaderSize;
    state_off_ = kHeaderSize + static_cast<size_t>(n_) * kCmdStride * 8;
  }

  ~ShmBridge()
  {
    unmap();
    close_fd();
  }

  ShmBridge(const ShmBridge &) = delete;
  ShmBridge & operator=(const ShmBridge &) = delete;

  const std::string & name() const { return name_; }
  uint32_t slots() const { return n_; }

  // Write `values` (kCmdStride per entry) into `slots` only, under one seqlock
  // write. The other components' slots are left exactly as they are.
  void write_command_slots(const std::vector<size_t> & slots, const double * values)
  {
    uint64_t s = u64(kCmdSeqOff);
    u64(kCmdSeqOff) = s + 1;  // odd: write in progress
    std::atomic_thread_fence(std::memory_order_release);
    for (size_t k = 0; k < slots.size(); ++k) {
      std::memcpy(
        f64(cmd_off_ + slots[k] * kCmdStride * 8), values + k * kCmdStride,
        kCmdStride * sizeof(double));
    }
    std::atomic_thread_fence(std::memory_order_release);
    u64(kCmdSeqOff) = s + 2;  // even: done
  }

  // Whole state block; false if the seqlock never converged.
  bool read_state(std::vector<double> & out)
  {
    const size_t n = static_cast<size_t>(n_) * kStateStride;
    out.resize(n);
    for (int spin = 0; spin < 1000; ++spin) {
      uint64_t s1 = u64(kStateSeqOff);
      if (s1 & 1ULL) { continue; }
      std::atomic_thread_fence(std::memory_order_acquire);
      std::memcpy(out.data(), f64(state_off_), n * sizeof(double));
      std::atomic_thread_fence(std::memory_order_acquire);
      uint64_t s2 = u64(kStateSeqOff);
      if (s1 == s2) { return true; }
    }
    return false;
  }

  // 0 until the servicer has published its first measured state.
  uint64_t state_seq() { return u64(kStateSeqOff); }
  bool state_ever_written() { return state_seq() != 0; }
  // Changes every time a servicer creates the segment.
  uint64_t epoch() { return u64(kEpochOff); }

private:
  void map(size_t size)
  {
    void * p = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (p == MAP_FAILED) {
      close_fd();
      throw std::runtime_error("actuo shm: mmap failed: " + path_);
    }
    base_ = static_cast<uint8_t *>(p);
    mapped_ = size;
  }

  void unmap()
  {
    if (base_ != nullptr) { ::munmap(base_, mapped_); }
    base_ = nullptr;
    mapped_ = 0;
  }

  void close_fd()
  {
    if (fd_ >= 0) { ::close(fd_); }
    fd_ = -1;
  }

  uint32_t & u32(size_t off) { return *reinterpret_cast<uint32_t *>(base_ + off); }
  uint64_t & u64(size_t off) { return *reinterpret_cast<uint64_t *>(base_ + off); }
  double * f64(size_t off) { return reinterpret_cast<double *>(base_ + off); }

  std::string name_;
  std::string path_;
  uint32_t n_ = 0;
  size_t size_ = 0;
  size_t mapped_ = 0;
  size_t cmd_off_ = 0;
  size_t state_off_ = 0;
  int fd_ = -1;
  uint8_t * base_ = nullptr;
};

}  // namespace actuo_hardware

#endif  // ACTUO_HARDWARE__SHM_BRIDGE_HPP_
