"""Inter-robot communication layer (proposal Table 1, Phase 6).

Implements all four Phase 6 properties behind neutral defaults:
  6.1 comm_range      — messages reach only robots within range (None = unlimited)
  6.2 packet_loss     — per-message drop probability (0.0 = reliable)
      comm_latency    — delivery delay in ticks (0 = same-tick)
  6.3 inbox / outbox  — FIFO queues per robot (by unique_id)
  6.4 comm_bandwidth  — max deliveries per robot per tick; excess is
                        DEFERRED to later ticks, not dropped (None = unlimited)

Semantic decisions (document these in the report):
- Single-hop network. Multi-hop information spread is NOT the network's
  job: it emerges from consensus gossip (CBBA relaying y/z/s content).
- Range and loss are evaluated at SEND time (the transmission moment);
  latency then delays arrival. Robots moving out of range mid-flight do
  not lose the message.
- Range metric: Euclidean distance between cell coordinates (radio is
  circular even on a grid). Swap `_in_range` to change.
- Payloads are deep-copied on send: robots share no memory, exactly as
  a real decentralised fleet. Mutating your dict after send() cannot
  retroactively change what others receive.
- Loss draws use the model's seeded RNG, so lossy runs are reproducible;
  with packet_loss == 0 no randomness is consumed, so default runs are
  bit-identical to a build without this module.

CBBA usage:
  Synchronous rounds inside allocate() (current style, ideal defaults):
      for each robot: robot.send(payload)                  # broadcast
      model.comms.flush_and_deliver(model.tick)            # exchange
      for each robot: msgs = robot.receive_all()
  Multi-tick decentralised rounds (Phase 6 proper): send during one
  allocate() call, read inboxes on the next tick — model.step() calls
  flush_and_deliver() once at the top of every tick.
"""

from __future__ import annotations

import copy
import itertools
import math
from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Message:
    sender: int                 # unique_id of the sending robot
    recipient: int              # unique_id of the receiver
    payload: dict = field(compare=False)
    sent_tick: int = 0
    deliver_tick: int = 0       # sent_tick + latency


class CommNetwork:
    """Owned by the model; robots talk only through it."""

    def __init__(self, model, comm_range: float | None = None,
                 packet_loss: float = 0.0, latency: int = 0,
                 bandwidth: int | None = None):
        self.model = model
        self.comm_range = comm_range
        self.packet_loss = packet_loss
        self.latency = int(latency)
        self.bandwidth = bandwidth
        self._seq = itertools.count()          # FIFO tie-breaker
        self._outboxes: dict[int, deque] = {}  # staged sends, per sender
        self._transit: list[tuple[int, int, Message]] = []  # in-flight
        self._pending: dict[int, deque] = {}   # due, awaiting bandwidth
        self._inboxes: dict[int, deque] = {}   # delivered, per recipient
        self.stats = {"sent": 0, "delivered": 0, "lost": 0,
                      "out_of_range": 0, "deferred": 0}

    # ------------------------------------------------------------- #
    # robot-facing API
    # ------------------------------------------------------------- #
    def send(self, sender, payload: dict, to=None) -> None:
        """Queue a message. `to` is a robot (or unique_id); None means
        broadcast to every robot currently in range."""
        recipients = self._resolve_recipients(sender, to)
        box = self._outboxes.setdefault(sender.unique_id, deque())
        for r in recipients:
            box.append((r.unique_id, copy.deepcopy(payload)))

    def receive_all(self, robot) -> list[Message]:
        """Drain and return the robot's inbox (FIFO)."""
        inbox = self._inboxes.get(robot.unique_id)
        if not inbox:
            return []
        out = list(inbox)
        inbox.clear()
        return out

    def neighbors(self, robot) -> list:
        """Robots currently within comm range (excluding self)."""
        return [r for r in self.model.robots
                if r is not robot and self._in_range(robot, r)]

    # ------------------------------------------------------------- #
    # model-facing API — called once at the top of every tick, and
    # explicitly by synchronous allocators between rounds
    # ------------------------------------------------------------- #
    def flush_and_deliver(self, now: int) -> None:
        self._flush_outboxes(now)
        self._deliver_due(now)

    # ------------------------------------------------------------- #
    def _flush_outboxes(self, now: int) -> None:
        for sender_id, box in self._outboxes.items():
            while box:
                recipient_id, payload = box.popleft()
                self.stats["sent"] += 1
                if self.packet_loss > 0.0 \
                        and self.model.random.random() < self.packet_loss:
                    self.stats["lost"] += 1
                    continue
                msg = Message(sender_id, recipient_id, payload,
                              sent_tick=now,
                              deliver_tick=now + self.latency)
                self._transit.append((msg.deliver_tick, next(self._seq), msg))

    def _deliver_due(self, now: int) -> None:
        due = [t for t in self._transit if t[0] <= now]
        self._transit = [t for t in self._transit if t[0] > now]
        for _, _, msg in sorted(due):
            self._pending.setdefault(msg.recipient, deque()).append(msg)
        for rid, queue in self._pending.items():
            quota = len(queue) if self.bandwidth is None else self.bandwidth
            inbox = self._inboxes.setdefault(rid, deque())
            while queue and quota > 0:
                inbox.append(queue.popleft())
                self.stats["delivered"] += 1
                quota -= 1
            if queue:
                self.stats["deferred"] += len(queue)

    def _resolve_recipients(self, sender, to) -> list:
        if to is None:
            return self.neighbors(sender)
        robot = to if hasattr(to, "unique_id") else \
            next(r for r in self.model.robots if r.unique_id == to)
        if self._in_range(sender, robot):
            return [robot]
        self.stats["out_of_range"] += 1
        return []

    def _in_range(self, a, b) -> bool:
        if self.comm_range is None:
            return True
        ax, ay = a.cell.coordinate
        bx, by = b.cell.coordinate
        return math.hypot(ax - bx, ay - by) <= self.comm_range

# --------------------------------------------------------------------- #
# Graph property of the comm network. Lives HERE, not in allocation.py:
# it needs only model.robots and model.comms, and keeping it in the
# allocator module forced MOACBBA.py -- which subclasses nothing from
# allocation but needs this bound -- to import allocation at module
# scope, which closed an import cycle
#   allocation -> MOACBBA -> allocation
# that only manifested when allocation was imported first.
# --------------------------------------------------------------------- #
def network_diameter(model) -> int:
    """D in Choi et al. Eq. (19): longest shortest path in the comm
    graph. Unlimited range is a complete graph, so D = 1. A disconnected
    graph has no finite diameter; fall back to the fleet size, which is
    the loosest bound that still terminates."""
    robots = model.robots
    n = len(robots)
    if n <= 1 or model.comms.comm_range is None:
        return 1
    adj = {r.robot_id: [b.robot_id for b in model.comms.neighbors(r)]
           for r in robots}
    best = 1
    for src in adj:
        seen = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in seen:
                    seen[v] = seen[u] + 1
                    q.append(v)
        if len(seen) < n:
            return n
        best = max(best, max(seen.values()))
    return max(1, best)