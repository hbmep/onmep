import os
import time
import json
from dataclasses import dataclass, asdict, field


def _now_ns() -> int:
    return time.perf_counter_ns()


@dataclass
class StepTimes:
    step: int
    t_propose_ns: int = 0
    t_step_ns: int = 0
    t_update_ns: int = 0
    t_ckpt_ns: int = 0
    t_iter_ns: int = 0
    agent_stop: bool = False
    tags: dict = field(default_factory=dict)


class Telemetry:
    """Buffered writer for per-iteration timings."""
    def __init__(
        self,
        out: str | None,
        mode: str = "ndjson",
        flush_every: int = 50,
        truncate_on_start: bool = True,
    ):
        self.out = out
        self.mode = mode
        self.flush_every = max(1, int(flush_every))
        self._buf: list[StepTimes] = []
        if self.out:
            os.makedirs(os.path.dirname(self.out), exist_ok=True)
            if truncate_on_start and self.mode == "ndjson":
                open(self.out, "w").close()
        return

    def add(self, st: StepTimes):
        self._buf.append(st)
        if self.out and self.mode == "ndjson" and len(self._buf) >= self.flush_every:
            self._flush_ndjson(append=True)
        return

    def close(self):
        if not self.out or not self._buf:
            self._buf.clear()
            return
        if self.mode == "ndjson":
            self._flush_ndjson(append=True)
        self._buf.clear()
        return

    def _flush_ndjson(self, append: bool):
        with open(self.out, "a" if append else "w") as f:
            for st in self._buf:
                f.write(json.dumps(asdict(st)) + "\n")
        self._buf.clear()
        return
