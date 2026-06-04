import hbmep as mep

from jax import random, numpy as jnp, vmap, Array
from jax.scipy.special import logsumexp

from onmep.agent.eig import EIG


class TTD(EIG):
    def __init__(
        self,
        *args,
        distance: str = "wasserstein-1",
        **kw,
    ):
        super().__init__(*args, **kw)
        self.name = "ttd"

        implemented = ("wasserstein-1", "wasserstein-2")
        if distance not in implemented:
            raise ValueError(f"distance must be in {implemented}, got {distance}")

        self.distance = distance
        return

    @property
    def distance_fn(self):
        match self.distance:
            case "wasserstein-1":
                return mep.metrics.wasserstein1
            case "wasserstein-2":
                return mep.metrics.wasserstein2
            case _:
                raise ValueError(f"Unknown TTD distance: {self.distance}")

    def ttd_one(
        self,
        key: Array,
        x: float,
        k: int,
        posterior: dict[str, jnp.ndarray],
        logw: jnp.ndarray | None = None,
        num_outer: int | None = None,
    ):
        """
        U(x, k) = sum_r E_y[ W(p(a_r | D, y, x, k), p(a_r | D)) ]
        where the expectation over y is approximated using outer Monte Carlo,
        and p(a_r | D, y, x) is approximated via importance reweighting of
        existing posterior draws.
        """
        a = posterior[mep.site.a]       # (S, K, R)
        a = a[:, k, :]                  # (S, R)
        S = a.shape[0]

        key, key_idx, key_y = random.split(key, 3)

        if logw is None:
            logw = jnp.zeros(S)
        elif logw.shape[0] != S:
            raise ValueError("logw and posterior sample size mismatch")

        logw = logw - logsumexp(logw)
        w = jnp.exp(logw)

        params_all = self._params_all(x, k, posterior)
        distance_fn = self.distance_fn

        def body_one_draw(kkey, i):
            params_s = [param[i] for param in params_all]
            y_s = self._sample_y(kkey, *params_s)

            loglik_all = vmap(
                lambda *params_j: self._loglik(y_s, *params_j)
            )(*params_all)

            # Importance weights for updated posterior p(theta | x, y):
            # w_post,j ∝ w_prior,j * p(y | theta_j, x)
            logw_post = logw + loglik_all
            logw_post = logw_post - logsumexp(logw_post)
            w_post = jnp.exp(logw_post)

            # Wasserstein distance
            vals_r = vmap(
                lambda r: distance_fn(a[:, r], a[:, r], weights1=w, weights2=w_post)
            )(jnp.arange(a.shape[1]))   # (R,)

            return vals_r

        if (num_outer is None) or (num_outer >= S):
            idx = jnp.arange(S)
            keys = random.split(key_y, S)
            vals = vmap(body_one_draw)(keys, idx)           # (S, R)
            val = jnp.sum(w[:, None] * vals, axis=0)        # (R,)
        else:
            idx = random.choice(key_idx, S, (num_outer,), replace=True, p=w)
            keys = random.split(key_y, num_outer)
            vals = vmap(body_one_draw)(keys, idx)           # (S, R)
            val = vals.mean(axis=0)                         # (R,)

        return jnp.sum(val)

    @mep.execute_on_gpu
    def _acquisition_fn(
        self,
        key,
        xs,
        ks,
        posterior,
        logw,
        num_outer=None,
        batch_size=64,
    ):
        """Return TTD vals with shape (K, N)."""
        N = xs.shape[0]
        keys_k = random.split(key, ks.shape[0])

        def eval_one_k(k_idx, kk):
            keys_x = random.split(kk, N)

            def one_x(xx, kx):
                return self.ttd_one(
                    kx,
                    xx,
                    k_idx,
                    posterior,
                    logw=logw,
                    num_outer=num_outer,
                )

            outs = []
            bs = N if batch_size is None else batch_size
            for start in range(0, N, bs):
                stop = min(start + bs, N)
                vals = vmap(one_x)(xs[start:stop], keys_x[start:stop])
                outs.append(vals)

            return jnp.concatenate(outs, axis=0)    # (N,)

        vals = vmap(eval_one_k)(ks, keys_k)         # (K, N)
        return vals

    def state_dict(self, include_posterior: bool = True) -> dict:
        d = super().state_dict(include_posterior=include_posterior)
        d["ttd"] = {"distance": self.distance}
        return d

    def load_state_dict(self, state: dict):
        super().load_state_dict(state)
        td = state.get("ttd", {})
        self.distance = td.get("distance", self.distance)
        return self
