import hbmep as mep
from jax import random, numpy as jnp, vmap, Array
from jax.scipy.special import logsumexp

from onmep.agent.eig import EIG


class EVR(EIG):
    def __init__(self, *args, target_parameter=mep.site.a, **kw):
        super().__init__(*args, **kw)
        self.name = "evr"
        self.target_parameter = target_parameter
        return

    @staticmethod
    def _prior_var(param: jnp.ndarray, w: jnp.ndarray):
        """
        param: (S, R)
        w: (S,) normalized weights
        """
        mean = jnp.sum(w[:, None] * param, axis=0)          # (R,)
        mean2 = jnp.sum(w[:, None] * (param ** 2), axis=0)  # (R,)
        return mean2 - mean ** 2                            # (R,)

    def eig_one(
        self,
        key: Array,
        x: float,
        k: int,
        posterior: dict[str, jnp.ndarray],
        logw: jnp.ndarray | None = None,
        num_outer: int | None = None,
    ):
        """
        U(x, k) = sum_r E_y[ Var_prior(a_r) - Var_post(a_r | y, x, k) ]
        where `a` is the target parameter and the expectation over y is
        approximated using outer Monte Carlo, and Var_post is computed
        via importance reweighting of existing draws.
        """
        a = posterior[self.target_parameter]    # (S, K, R)
        a = a[:, k, :]                          # (S, R)
        S = a.shape[0]

        key, key_idx, key_y = random.split(key, 3)

        if logw is None:
            logw = jnp.zeros(S)
        elif logw.shape[0] != S:
            raise ValueError("logw and posterior sample size mismatch")

        logw = logw - logsumexp(logw)
        w = jnp.exp(logw)

        # Prior variance
        var_prior = self._prior_var(a, w=w)                 # (R,)
        params_all = self._params_all(x, k, posterior)      # (S, R) each

        def body_one_draw(kkey, i):
            params_s = [param[i] for param in params_all]   # (R,) each
            y_s = self._sample_y(kkey, *params_s)           # (R,)

            loglik_all = vmap(
                lambda *params_j: self._loglik(y_s, *params_j)
            )(*params_all)  # (S,)

            # Importance weights for updated posterior p(theta | x, y):
            # w_post,j ∝ w_prior,j * p(y | theta_j, x)
            logw_post = logw + loglik_all
            logw_post = logw_post - logsumexp(logw_post)
            w_post = jnp.exp(logw_post)                     # (S,)

            var_post = self._prior_var(a, w=w_post)         # (R,)
            return var_prior - var_post                     # (R,)

        if (num_outer is None) or (num_outer >= S):
            idx = jnp.arange(S)
            keys = random.split(key_y, S)
            reductions = vmap(body_one_draw)(keys, idx)             # (S, R)
            reduction = jnp.sum(w[:, None] * reductions, axis=0)    # (R,)
        else:
            idx = random.choice(key_idx, S, (num_outer,), replace=True, p=w)
            keys = random.split(key_y, num_outer)
            reductions = vmap(body_one_draw)(keys, idx)             # (S, R)
            reduction = reductions.mean(axis=0)                     # (R,)

        return jnp.sum(reduction)

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
