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
        x: float,
        k: int,
        posterior: dict[str, jnp.ndarray],
        logw: jnp.ndarray | None = None,
        num_outer: int | None = None,
    ):
        """
        Monte Carlo MI estimate for candidate action (x, k):
            E_{θ,y}[ log p(y|θ,x,k) - log p(y|x,k) ]
        """
        S = posterior[mep.site.a].shape[0]
        key, key_idx, key_y = random.split(key, 3)

        if logw is None:
            logw = jnp.zeros(S)
        elif logw.shape[0] != S:
            raise ValueError("logw and posterior sample size mismatch")

        logw = logw - logsumexp(logw)
        w = jnp.exp(logw)

        params_all = self._params_all(x, k, posterior)      # (S, R) each

        def body_one_draw(kkey, i):
            params_s = [param[i] for param in params_all]   # (R,) each
            y_s = self._sample_y(kkey, *params_s)           # (R,)
            loglik_s = self._loglik(y_s, *params_s)         # ()
            loglik_all = vmap(
                lambda *params_j: self._loglik(y_s, *params_j)
            )(*params_all)  # (S,)
            log_pred = logsumexp(loglik_all + logw)
            return loglik_s - log_pred

        if (num_outer is None) or (num_outer >= S):
            idx = jnp.arange(S)
            keys = random.split(key_y, S)
            gains = vmap(body_one_draw)(keys, idx)          # (S,)
            gain = jnp.sum(w * gains)                       # ()
        else:
            idx = random.choice(key_idx, S, (num_outer,), replace=True, p=w)
            keys = random.split(key_y, num_outer)
            gains = vmap(body_one_draw)(keys, idx)          # (num_outer,)
            gain = gains.mean()                             # ()

        return gain

    @mep.execute_on_gpu
    def _acquisition_fn(
        self,
        key,
        xs,
        ks,
        posterior,
        logw,
        num_outer=None,
        batch_size=64
    ):
        """Return acquisition vals with shape (K, N)."""
        N = xs.shape[0]
        keys_k = random.split(key, ks.shape[0])

        def eval_one_k(k_idx, kk):
            keys_x = random.split(kk, N)

            def one_x(xx, kx):
                return self.eig_one(
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

    def _compute_acquisition(self, *, feature0=None, response=None):
        self._key, sub = random.split(self._key)

        if self.posterior is None:
            raise ValueError("No posterior available for acquisition.")

        posterior = self.posterior
        logw = self.logw

        if response is not None:
            posterior = {
                u: v[..., response:response + 1]
                for u, v in posterior.items()
            }

        if feature0 is None:
            ks = self.grid_f
        else:
            ks = jnp.asarray([feature0], dtype=self.grid_f.dtype)

        vals_sub = self._acquisition_fn(
            sub,
            self.grid_x,
            ks,
            posterior,
            logw,
            num_outer=self.num_outer,
            batch_size=self.batch_size,
        )

        if feature0 is None:
            return vals_sub

        vals = jnp.full((self.K, self.grid_x.shape[0]), -jnp.inf)
        vals = vals.at[feature0, :].set(vals_sub[0])
        return vals

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
