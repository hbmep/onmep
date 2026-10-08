import hbmep as mep
from jax import random, numpy as jnp, vmap, Array
from jax.scipy.special import logsumexp

from onmep.agent.eig import EIG


class EVR(EIG):
    def __init__(
        self,
        *args,
        target_parameter: str = mep.site.a,
        **kw
    ):
        super().__init__(*args, **kw)
        self.name = "evr"
        self._target_param = target_parameter
        self._target_idx = None
        return

    def _on_model_changed(self):
        super()._on_model_changed()
        self._target_idx = None
        return

    @property
    def target_parameter(self):
        return self._target_param

    @target_parameter.setter
    def target_parameter(self, val: str):
        self._target_param = val
        self._target_idx = None
        return

    @property
    def target_idx(self):
        if self._target_idx is None:
            self._target_idx = self._args.index(self.target_parameter)
        return self._target_idx

    @staticmethod
    def _prior_var(param: jnp.ndarray, w: jnp.ndarray):
        """
        param: (S,)
        w: (S,), normalized
        """
        mean = jnp.sum(w * param)                   # ()
        mean2 = jnp.sum(w * (param ** 2), axis=0)   # ()
        return mean2 - mean ** 2                    # ()
    
    def eig_one(
        self,
        key: Array,
        x: float,                           # ()
        posterior: jnp.ndarray,             # (P, S)
        logw: jnp.ndarray,                  # (S,), normalized
        num_outer: int | None = None,       # outer MC sample size
    ):
        """
        Monte Carlo MI estimate for candidate action
            E_{y|D,x}[ Var_prior(a) - Var_post(a|x,y) ]
        Var_post is computed via importance reweighting of existing draws.
        """
        S = posterior.shape[1]
        key, key_idx, key_y = random.split(key, 3)

        params = self._params_all(x, posterior)                 # (S,) each
        w = jnp.exp(logw)                                       # (S,)

        a = posterior[self.target_idx]                          # (S,)

        # Prior variance
        var_prior = self._prior_var(a, w=w)                     # ()

        def body_one_draw(kkey, i):
            params_s = [param[i] for param in params]           # () each
            y_s = self._sample_y(kkey, *params_s)               # ()

            loglik_all = vmap(
                lambda *params_j: self._loglik(y_s, *params_j)
            )(*params)                                          # (S,)
            # Importance weights for updated posterior p(theta | y):
            # w_post,j ∝ w_prior,j * p(y | theta_j)
            logw_post = logw + loglik_all                       # (S,)
            logw_post = logw_post - logsumexp(logw_post)        # (S,)
            w_post = jnp.exp(logw_post)                         # (S,)

            var_post = self._prior_var(a, w=w_post)             # (R,)
            return var_prior - var_post                         # (R,)

        if (num_outer is None) or (num_outer >= S):
            idx = jnp.arange(S)
            keys = random.split(key_y, S)
            reductions = vmap(body_one_draw)(keys, idx)         # (S,)
            reduction = jnp.sum(w * reductions, axis=0)         # ()
        else:
            idx = random.choice(key_idx, S, (num_outer,), replace=True, p=w)
            keys = random.split(key_y, num_outer)
            reductions = vmap(body_one_draw)(keys, idx)         # (num_outer,)
            reduction = jnp.mean(reductions)                    # ()

        return reduction                                        # ()

    def state_dict(self, include_posterior: bool = True) -> dict:
        d = super().state_dict(include_posterior=include_posterior)
        d["evr"] = {
            "target_parameter": self.target_parameter,
        }
        return d

    def load_state_dict(self, state: dict):
        super().load_state_dict(state)
        ed = state.get("evr", {})
        self.target_parameter = ed.get("target_parameter", self.target_parameter)
        return self
