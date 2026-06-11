from dataclasses import replace

import jax.numpy as jnp

from onmep import Observation
from onmep.agent import Action, Proposal


class RoundRobinResponseMixin:
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        protocol = "RoundRobinResponseMixin"

        if not self.num_response > 1:
            raise RuntimeError(f"{protocol} not needed for only one response.")

        if self.infer != "mcmc":
            raise NotImplementedError(f"{protocol} only supports infer=mcmc.")

        if self.K != 1:
            raise NotImplementedError(f"{protocol} only supports K=1.")

        self._rr_response_idx: int = 0
        self._rr_update_mode: str = "round-robin"   # "round-robin", "joint", "independent"
        self._mask_non_target: bool = False         # treat responses as iterative
        return

    def update(
        self,
        obs: Observation,
        *,
        feature0: int | None = None,
        response: int | None = None,
    ):
        was_postwarm = self._in_postwarm_phase

        if was_postwarm and self._mask_non_target:
            mask = jnp.arange(obs.response.shape[0]) != self._rr_response_idx
            response = jnp.asarray(obs.response).at[mask].set(jnp.nan)
            obs_masked = replace(obs, response=response)
            self.append_obs(obs_masked)
        else:
            self.append_obs(obs)

        if was_postwarm:
            self._rr_response_idx += 1
            self._rr_response_idx %= self.num_response

        if self._last_proposal is not None:
            self._last_proposal.is_pending = False

        should_update = self._should_update()
        if should_update:
            match self._rr_update_mode:
                case "round-robin":
                    self._update_posterior(response=self._rr_response_idx)

                case "joint":
                    self._update_posterior(response=None)

                case "independent":
                    for response_idx in range(self.num_response):
                        self._update_posterior(response=response_idx)

                case _:
                    raise ValueError(f"Invalid update mode {self._rr_update_mode}")

        return self.should_stop()

    def propose(
        self,
        *,
        feature0: int | None = None,
        response: int | None = None
    ) -> Action:
        use_last = (
            self._last_proposal is not None
            and self._last_proposal.is_pending
        )

        if use_last:
            return self._last_proposal.action

        if self._in_zero_phase:
            phase = "zero"
            shared = self.share_zero
            action = self._zero_propose()

        elif self._in_warm_phase:
            phase = "warm"
            shared = self.share_warm
            action = self._uniform_propose()

        elif self.use_acquisition:
            phase = "postwarm"
            shared = False
            action = self._propose(response=self._rr_response_idx)

        else:
            phase = "postwarm"
            shared = False
            action = self._uniform_propose()

        self._last_proposal = Proposal(
            action=action, phase=phase, shared=shared, is_pending=True
        )
        return action

    def state_dict(self, include_posterior=True):
        d = super().state_dict(include_posterior=include_posterior)
        d["round_robin_response"] = {
            "_rr_response_idx": self._rr_response_idx,
            "_rr_update_mode": self._rr_update_mode,
            "_mask_non_target": self._mask_non_target,
        }
        return d

    def load_state_dict(self, state):
        super().load_state_dict(state)
        d = state.get("round_robin_response", {})
        self._rr_response_idx = d.get("_rr_response_idx", self._rr_response_idx)
        self._rr_update_mode = d.get("_rr_update_mode", self._rr_update_mode)
        self._mask_non_target = d.get("_mask_non_target", self._mask_non_target)
        return self


def make_round_robin_response(agent_cls):
    class RoundRobinResponse(RoundRobinResponseMixin, agent_cls):
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)

    RoundRobinResponse.__name__ = f"RoundRobinResponse{agent_cls.__name__}"
    RoundRobinResponse.__qualname__ = RoundRobinResponse.__name__
    return RoundRobinResponse
