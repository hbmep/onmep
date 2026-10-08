import hbmep as mep
from jax import random, numpy as jnp, vmap, Array
from jax.scipy.special import logsumexp

from onmep.agent.base_agent import BaseAgent


class EIG(BaseAgent):
    def __init__(
        self,
        *args,
        num_outer: int | None = None,
        batch_size: int | None = 51,
        **kw,
    ):
        super().__init__(*args, **kw)
        self.name = "eig"
        self.num_outer = num_outer
        self.batch_size = batch_size
        return

    def eig_one(
        self,
        key: Array,
        x: float,                           # ()
        posterior: jnp.ndarray,             # (P, S)
        logw: jnp.ndarray,                  # (S,), normalized
        num_outer: int | None = None,       # outer MC sample size
    ):
        """
        Monte Carlo MI estimate for candidate action:
            E_{y,θ|D,x}[ log p(y|θ,x) - log p(y|D,x) ]
        """
        S = posterior.shape[1]
        key, key_idx, key_y = random.split(key, 3)

        params = self._params_all(x, posterior)                 # (S,) each
        w = jnp.exp(logw)                                       # (S,)

        def body_one_draw(kkey, i):
            params_s = [param[i] for param in params]           # () each
            y_s = self._sample_y(kkey, *params_s)               # ()
            loglik_s = self._loglik(y_s, *params_s)             # ()
            loglik_all = vmap(
                lambda *params_j: self._loglik(y_s, *params_j)
            )(*params)                                          # (S,)
            log_pred = logsumexp(loglik_all + logw, axis=0)     # ()
            return loglik_s - log_pred                          # ()

        if (num_outer is None) or (num_outer >= S):
            idx = jnp.arange(S)
            keys = random.split(key_y, S)
            gains = vmap(body_one_draw)(keys, idx)              # (S,)
            gain = jnp.sum(w * gains)                           # ()
        else:
            idx = random.choice(key_idx, S, (num_outer,), replace=True, p=w)
            keys = random.split(key_y, num_outer)
            gains = vmap(body_one_draw)(keys, idx)              # (num_outer,)
            gain = jnp.mean(gains)                              # ()

        return gain                                             # ()

    @mep.execute_on_gpu
    def _acquisition_fn(
        self,
        key: Array,
        xs: jnp.ndarray,                # (G,) grid points
        ks: jnp.ndarray,                # ks ⊆ jnp.arange(K)
        rs: jnp.ndarray,                # rs ⊆ jnp.arange(R)
        posterior: jnp.ndarray,         # (P, S, K, R)
        logw: jnp.ndarray,              # (S, K, R), normalized
        num_outer: int | None = None,   # outer MC sample size
        batch_size: int = 64,           # batch size across grid
    ):
        """
        Return acquisition values with shape (num_ks, G), where
        xs: (G,)
        ks: (num_ks,)
        rs: (num_rs,)
        posterior: (P, S, K, R)
        logw: (S, K, R)
        """
        G = xs.shape[0]
        num_ks = ks.shape[0]
        num_rs = rs.shape[0]

        if logw.shape != posterior[0].shape:
            raise ValueError("logw and posterior sample size mismatch")

        keys_k = random.split(key, num_ks)

        def one_k(k, key_k):
            def one_x(x, key_x):
                def one_r(r, key_r):
                    return self.eig_one(
                        key=key_r,
                        x=x,
                        posterior=posterior[..., k, r],
                        logw=logw[..., k, r],
                        num_outer=num_outer,
                    )
                
                keys_r = (
                    key_x[None]
                    if num_rs == 1
                    else random.split(key_x, num_rs)
                )                                               # (num_rs, 2)
                vals = vmap(one_r)(rs, keys_r)                  # (num_rs,)
                return jnp.sum(vals)                            # ()

            outs = []
            bs = G if batch_size is None else batch_size
            keys_x = random.split(key_k, G)

            for start in range(0, G, bs):
                stop = min(start + bs, G)
                vals = vmap(one_x)(
                    xs[start:stop],
                    keys_x[start:stop],
                )                                               # (bs,)
                outs.append(vals)

            return jnp.concatenate(outs, axis=0)                # (G,)

        vals = vmap(one_k)(ks, keys_k)                          # (num_k, G)
        return vals

    def _compute_acquisition(self, *, feature0=None, response=None):
        """Return acquisition values with shape (K, G)."""
        self._key, sub = random.split(self._key)

        if self.posterior is None:
            raise ValueError("No posterior available for acquisition.")

        xs = self.grid_x
        ks = (
            self.grid_f if feature0 is None
            else jnp.asarray([feature0], dtype=self.grid_f.dtype)
        )
        rs = (
            self.grid_r if response is None
            else jnp.asarray([response], dtype=jnp.int32)
        )

        vals_sub = self._acquisition_fn(
            key=sub,
            xs=xs,
            ks=ks,
            rs=rs,
            posterior=self.posterior.array[:len(self._args)],
            logw=self.posterior.logw,
            num_outer=self.num_outer,
            batch_size=self.batch_size,
        )       # (num_ks, G)

        if feature0 is None:
            return vals_sub

        vals = jnp.full((self.grid_f.shape[0], self.grid_x.shape[0]), -jnp.inf)     # (K, G)
        vals = vals.at[feature0, :].set(vals_sub[0])
        return vals                                                                 # (K, G)

    def state_dict(self, include_posterior: bool = True) -> dict:
        d = super().state_dict(include_posterior=include_posterior)
        d["eig"] = {
            "num_outer": self.num_outer,
            "batch_size": self.batch_size,
        }
        return d

    def load_state_dict(self, state: dict):
        super().load_state_dict(state)
        ed = state.get("eig", {})
        self.num_outer = ed.get("num_outer", self.num_outer)
        self.batch_size = ed.get("batch_size", self.batch_size)
        return self
