// actuo_uart_relay: put a Kondo RCB-4mini on WiFi with an Arduino UNO R4 WiFi.
//
// The host (actuo_bus_node with the kondo_rcb4_udp transport) sends each of
// the board's packets in a UDP datagram; this sketch writes it to Serial1
// (D1 = TX, D0 = RX, 8E1, wired to the RCB-4mini's COM port), reads the
// board's answer and sends it back in one datagram. It knows where a packet
// ends -- the first byte is its length -- and nothing else about it: servos,
// angles and opcodes stay in the host's bus.
//
// The datagrams are actuo/drivers/uart_relay_protocol.py; the same logic in
// Python, for testing without this board, is actuo/drivers/uart_relay.py.
// Keep the three in step.
//
// Failsafe: the host gives this sketch packets (S) to send by itself when the
// host has been silent for silence_ms -- the wheels stopped, so a robot whose
// laptop left the network does not drive off. Sent once per silence; counted
// and reported back in every answer.
//
// Build:
//   arduino-cli compile --fqbn arduino:renesas_uno:unor4wifi actuo_uart_relay
// Before that, copy arduino_secrets.h.example to arduino_secrets.h and put the
// network's name and password in it (arduino_secrets.h is not committed).
//
// Wiring and level: the UNO R4 is a 5 V board; the RCB-4mini COM port's
// levels are NOT verified here. Check before
// connecting.

#include <WiFiS3.h>

#include "arduino_secrets.h"

// -- configuration -----------------------------------------------------------

static const unsigned int kPort = 9750;          // kondo_rcb4_udp DEFAULT_PORT
static const unsigned long kInitialBaud = 115200;  // the board at power-up

// -- protocol (uart_relay_protocol.py) ----------------------------------------

static const uint8_t kForward = 'F';
static const uint8_t kBaud = 'B';
static const uint8_t kFailsafe = 'S';
static const uint8_t kPing = 'P';
static const uint8_t kReply = 'R';
static const uint8_t kAck = 'A';
static const uint8_t kStatusOk = 0;
static const uint8_t kStatusTimeout = 1;
static const uint8_t kStatusBadRequest = 2;

static const size_t kMaxPacket = 255;          // the length byte's limit
static const size_t kMaxFailsafeBytes = 512;   // MAX_FAILSAFE_BYTES
static const size_t kHeader = 3;               // type, seq
static const size_t kAnswerHeader = 6;         // type, seq, status, failsafes

// -- state -------------------------------------------------------------------

WiFiUDP udp;
uint8_t request[kHeader + 2 + kMaxFailsafeBytes];
uint8_t answer[kAnswerHeader + kMaxPacket];
uint8_t failsafePackets[kMaxFailsafeBytes];
size_t failsafeLength = 0;
unsigned long failsafeSilenceMs = 0;
uint16_t failsafes = 0;
bool fired = false;
unsigned long lastHeardMs = 0;
uint32_t baud = kInitialBaud;

static uint16_t readU16(const uint8_t *p) { return p[0] | (uint16_t(p[1]) << 8); }

static uint32_t readU32(const uint8_t *p) {
  return p[0] | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) |
         (uint32_t(p[3]) << 24);
}

// One length-prefixed packet from the board into buffer; its length, or 0 if
// it did not complete within timeoutMs.
static size_t readPacket(uint8_t *buffer, unsigned long timeoutMs) {
  unsigned long start = millis();
  size_t have = 0;
  while (millis() - start < timeoutMs) {
    while (Serial1.available() > 0 && have < kMaxPacket) {
      buffer[have++] = uint8_t(Serial1.read());
      if (buffer[0] == 0) {
        return 0;
      }
      if (have >= buffer[0]) {
        return have;
      }
    }
  }
  return 0;
}

static size_t transact(const uint8_t *packet, size_t length, uint8_t *reply,
                       unsigned long timeoutMs) {
  while (Serial1.available() > 0) {  // a late answer is not this one's
    Serial1.read();
  }
  Serial1.write(packet, length);
  Serial1.flush();
  return readPacket(reply, timeoutMs);
}

static void sendAnswer(uint8_t kind, uint16_t seq, uint8_t status,
                       size_t packetLength) {
  answer[0] = kind;
  answer[1] = seq & 0xFF;
  answer[2] = seq >> 8;
  answer[3] = status;
  answer[4] = failsafes & 0xFF;
  answer[5] = failsafes >> 8;
  udp.beginPacket(udp.remoteIP(), udp.remotePort());
  udp.write(answer, kAnswerHeader + packetLength);
  udp.endPacket();
}

// Whether body holds whole length-prefixed packets, back to back.
static bool wholePackets(const uint8_t *body, size_t length) {
  size_t i = 0;
  while (i < length) {
    if (body[i] == 0 || i + body[i] > length) {
      return false;
    }
    i += body[i];
  }
  return true;
}

static void handle(size_t length) {
  if (length < kHeader) {
    return;
  }
  uint8_t kind = request[0];
  uint16_t seq = readU16(request + 1);
  const uint8_t *body = request + kHeader;
  size_t bodyLength = length - kHeader;
  lastHeardMs = millis();
  fired = false;
  if (kind == kForward && bodyLength >= 3) {
    size_t got = transact(body + 2, bodyLength - 2, answer + kAnswerHeader,
                          readU16(body));
    sendAnswer(kReply, seq, got ? kStatusOk : kStatusTimeout, got);
  } else if (kind == kBaud && bodyLength == 4) {
    uint32_t rate = readU32(body);
    if (rate != baud) {  // reopened only when it changes, as uart_relay.py
      Serial1.end();
      Serial1.begin(rate, SERIAL_8E1);
      baud = rate;
    }
    sendAnswer(kAck, seq, kStatusOk, 0);
  } else if (kind == kFailsafe && bodyLength >= 2 &&
             bodyLength - 2 <= kMaxFailsafeBytes &&
             wholePackets(body + 2, bodyLength - 2)) {
    failsafeSilenceMs = readU16(body);
    failsafeLength = bodyLength - 2;
    memcpy(failsafePackets, body + 2, failsafeLength);
    Serial.print("failsafe: ");
    Serial.print(failsafeLength);
    Serial.print(" bytes after ");
    Serial.print(failsafeSilenceMs);
    Serial.println(" ms of silence");
    sendAnswer(kAck, seq, kStatusOk, 0);
  } else if (kind == kPing) {
    sendAnswer(kAck, seq, kStatusOk, 0);
  } else {
    sendAnswer(kAck, seq, kStatusBadRequest, 0);
  }
}

static void watchdog() {
  if (fired || failsafeLength == 0 || failsafeSilenceMs == 0 ||
      millis() - lastHeardMs < failsafeSilenceMs) {
    return;
  }
  fired = true;
  failsafes++;
  uint8_t scratch[kMaxPacket];
  size_t i = 0;
  while (i < failsafeLength) {
    transact(failsafePackets + i, failsafePackets[i], scratch, 50);
    i += failsafePackets[i];
  }
  digitalWrite(LED_BUILTIN, HIGH);  // stays on: the host went away once
  Serial.println("FAILSAFE sent: the host was silent");
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(115200);
  Serial1.begin(kInitialBaud, SERIAL_8E1);
  if (WiFi.status() == WL_NO_MODULE) {
    Serial.println("no WiFi module answers; stopping");
    while (true) {
    }
  }
  while (WiFi.begin(SECRET_SSID, SECRET_PASS) != WL_CONNECTED) {
    Serial.println("joining the network failed; retrying in 2 s");
    delay(2000);
  }
  udp.begin(kPort);
  Serial.print("actuo_uart_relay at ");
  Serial.print(WiFi.localIP());
  Serial.print(":");
  Serial.println(kPort);
  lastHeardMs = millis();
}

void loop() {
  int length = udp.parsePacket();
  if (length > 0) {
    size_t n = udp.read(request, sizeof(request));
    handle(n);
  }
  watchdog();
}
