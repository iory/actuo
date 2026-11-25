"""Who answers on a request/response bus: the bookkeeping, not the wire.

A bus whose servos answer when asked (FEETECH, DYNAMIXEL) decides presence the
same way whatever the protocol:

* a servo is **present** once it has answered (and passed whatever check the
  bus makes of a servo that comes back); it becomes **absent** after
  ``absent_after_misses`` cycle reads in a row without its reply -- one lost
  packet is not a missing module;
* an absent servo is left out of the cycle read (kept in, every cycle would
  wait out the read timeout for it, so a robot would run slower the more
  modules were missing) and instead, one id per turn, every
  ``presence_period`` cycle reads, absent servos and watched ids are probed
  with one short request;
* watched ids (:meth:`actuo.core.contracts.Bus.watch`) are only ever probed.

:class:`PresenceTracker` keeps that state; the bus says what happened on the
wire (:meth:`~PresenceTracker.answered`, :meth:`~PresenceTracker.missed`,
:meth:`~PresenceTracker.watched_answered`) and asks it whom to probe next
(:meth:`~PresenceTracker.next_probe`). The probes themselves -- a PING, a read
of the model number -- are the bus's.

A bus that does not ask (the RCB-4mini, whose board polls the servos itself;
CAN drives that upload on their own) decides presence from what arrives, and
keeps its own bookkeeping behind the same contract.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

#: Cycle reads in a row without a reply that make a hot-plug servo absent.
DEFAULT_ABSENT_AFTER_MISSES = 3
#: Cycle reads between two probes (one absent servo or watched id per probe).
DEFAULT_PRESENCE_PERIOD = 10

ABSENT = "absent"
WATCHED = "watched"


class PresenceTracker:
    """Presence of one bus's servos and of the ids it watches.

    Parameters
    ----------
    own_ids : sequence
        The bus's servos.
    hot_plug : bool
        Servos may be absent and come and go. Without it a servo is present
        once it has answered and is never taken away (misses are still
        counted, for diagnostics).
    absent_after_misses : int
        With ``hot_plug``, cycle reads in a row without a reply that make a
        servo absent.
    presence_period : int
        Cycle reads between two probes.
    watch_ids : sequence
        Ids that are not the bus's servos but whose presence is wanted.
    """

    def __init__(self, own_ids: Sequence, hot_plug: bool = False,
                 absent_after_misses: int = DEFAULT_ABSENT_AFTER_MISSES,
                 presence_period: int = DEFAULT_PRESENCE_PERIOD,
                 watch_ids: Sequence = ()) -> None:
        if int(absent_after_misses) < 1 or int(presence_period) < 1:
            raise ValueError("absent_after_misses and presence_period must be >= 1")
        self.own_ids = list(own_ids)
        self.hot_plug = bool(hot_plug)
        self.absent_after_misses = int(absent_after_misses)
        self.presence_period = int(presence_period)
        self._own = set(self.own_ids)
        self._present: Dict = {}
        self._absent_why: Dict = {}
        self._reported: Set = set()
        self._misses: Dict = {}
        self._watched: Dict = {}
        self._watch_ids: Tuple = ()
        self._reads = 0
        self._turn = 0
        self.set_watch_ids(watch_ids)

    # --- watched ids ---------------------------------------------------------
    @property
    def watch_ids(self) -> Tuple:
        return self._watch_ids

    def set_watch_ids(self, ids: Iterable) -> None:
        """Replace the watched ids (the bus's own servos are left out)."""
        self._watch_ids = tuple(sorted(set(ids) - self._own))
        self._watched = {i: v for i, v in self._watched.items() if i in self._watch_ids}

    def watched_answered(self, device_id, answered: bool) -> None:
        """Record a watched id's answer (or silence) to its probe."""
        self._watched[device_id] = bool(answered)

    def watched(self, device_id) -> bool:
        """Whether a watched id answered its last probe (False before the first)."""
        return self._watched.get(device_id, False)

    # --- own servos ----------------------------------------------------------
    def present(self, device_id) -> bool:
        if self.hot_plug:
            return self._present.get(device_id, False)
        return device_id in self._reported

    def absence_reason(self, device_id) -> str:
        if self.present(device_id):
            return ""
        return self._absent_why.get(device_id, "never answered")

    def mark_present(self, device_id) -> None:
        """A servo answered and passed the bus's check."""
        self._present[device_id] = True
        self._absent_why.pop(device_id, None)
        self._reported.add(device_id)
        self._misses[device_id] = 0

    def mark_absent(self, device_id, why: str) -> None:
        self._present[device_id] = False
        self._absent_why[device_id] = why

    def answered(self, device_id) -> None:
        """A servo answered the cycle read."""
        self._reported.add(device_id)
        self._misses[device_id] = 0

    def missed(self, device_id) -> int:
        """A polled servo did not answer the cycle read; returns the streak.

        With ``hot_plug`` the streak reaching ``absent_after_misses`` makes it
        absent.
        """
        streak = self._misses.get(device_id, 0) + 1
        self._misses[device_id] = streak
        if self.hot_plug and streak >= self.absent_after_misses:
            self.mark_absent(device_id, "no reply to {} reads in a row".format(streak))
        return streak

    def consecutive_misses(self, device_id) -> int:
        return self._misses.get(device_id, 0)

    def polled(self, ids: Iterable) -> List:
        """Of ``ids``, those the cycle read asks: all, or with ``hot_plug``
        only the present ones."""
        return [i for i in ids if not self.hot_plug or self.present(i)]

    # --- probes --------------------------------------------------------------
    def next_probe(self) -> Optional[Tuple[str, object]]:
        """Called once per cycle read: ``(ABSENT, id)`` or ``(WATCHED, id)``
        when a probe is due this cycle, else ``None``."""
        self._reads += 1
        if self._reads % self.presence_period:
            return None
        turn = [(ABSENT, i) for i in self.own_ids if self.hot_plug and not self.present(i)]
        turn += [(WATCHED, i) for i in self._watch_ids]
        if not turn:
            return None
        item = turn[self._turn % len(turn)]
        self._turn += 1
        return item

    def answering_ids(self) -> List:
        """Present servos and watched ids that answered their last probe."""
        ids = {i for i in self.own_ids if self.present(i)}
        ids.update(i for i, seen in self._watched.items() if seen)
        return sorted(ids)
