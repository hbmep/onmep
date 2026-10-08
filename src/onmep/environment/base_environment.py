import logging
from abc import ABC

import hbmep as mep
from onmep import (
    CheckpointManager,
    Observation,
    Telemetry,
    StepTimes,
    _now_ns,
)
from onmep.agent.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class BaseEnvironment(ABC):
    def __init__(self, *, lo=float("-inf"), hi=float("inf")):
        self.name = "base_environment"
        self.lo = lo
        self.hi = hi

    def should_stop(self) -> bool:
        return False

    def _step(self, intensity: float, features: list[int]) -> Observation:
        raise NotImplementedError

    def step(self, action) -> Observation:
        return self._step(intensity=action.x, features=action.f)

    def state_dict(self) -> dict:
        return {
            "name": self.name,
            "lo": self.lo,
            "hi": self.hi,
        }

    def load_state_dict(self, state: dict):
        self.name = state.get("name", self.name)
        self.lo = state.get("lo", self.lo)
        self.hi = state.get("hi", self.hi)
        return self

    @mep.timing
    def interact(
        self,
        *,
        agent: BaseAgent,
        target: int,                            # target number of observations
        ckpt: CheckpointManager | None = None,
        save_every: int = 1,
        timings_out: str | None = None,         # e.g., "/.../reports/run_123/timings.ndjson"
        timings_flush_every: int = 1,           # flush buffer every N steps (ndjson)
        timings_overwrite: bool = True,
        timings_tags: dict | None = None,       # attached to each row (e.g., {"agent":"EIG", "draw":7})
        update_on_action_feature: bool = False,
    ):
        """
        Run until agent.data.n == target.
        Checkpoint at every `save_every` iteration.
        """
        if agent.data.n >= target:
            logging.info(
                f"Already at {agent.data.n} ≥ target {target} observations. "
                "Nothing to do."
            )
            return

        tel = Telemetry(
            out=timings_out,
            flush_every=timings_flush_every,
            truncate_on_start=timings_overwrite,
        ) if timings_out else None
        tags = dict(timings_tags or {})

        obs = None
        agent_stop = False

        try:
            # Episode loop
            while agent.data.n < target:
                t_iter0 = _now_ns()

                # Agent update
                t0 = _now_ns()
                if obs is not None:
                    update_kw = {}

                    if update_on_action_feature:
                        update_kw["feature0"] = obs.features[0]

                    agent_stop = agent.update(obs, **update_kw)

                st = StepTimes(step=agent.data.n, tags=tags.copy())
                st.t_update_ns = _now_ns() - t0
                st.agent_stop = agent_stop

                # Agent propose
                t0 = _now_ns()
                action = agent.propose()
                st.t_propose_ns = _now_ns() - t0

                # Environment step to obtain new observation
                t0 = _now_ns()
                obs = self.step(action)
                st.t_step_ns = _now_ns() - t0

                # Checkpoint
                if ckpt and (agent.data.n % save_every == 0):
                    t0 = _now_ns()
                    ckpt.save(
                        step=agent.data.n,
                        agent_state=agent.state_dict(),
                        env_state=self.state_dict(),
                        extra_meta={
                            "target": target,
                            "tags": tags,
                            "obs": obs.state_dict() if obs is not None else None,
                            "action": action.state_dict() if action is not None else None,
                        },
                    )
                    st.t_ckpt_ns = _now_ns() - t0

                st.t_iter_ns = _now_ns() - t_iter0
                logger.info(
                    f"num_samples: {agent.data.n}/{target}, "
                    f"update: {st.t_update_ns / 1_000_000_000:.2f}s, "
                    f"propose: {st.t_propose_ns / 1_000_000_000:.2f}s, "
                    f"iter: {st.t_iter_ns / 1_000_000_000:.2f}s, "
                )

                if tel:
                    tel.add(st)

                if agent_stop or self.should_stop():
                    break

        finally:
            if tel:
                tel.close()

        return
