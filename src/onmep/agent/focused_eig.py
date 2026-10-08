import hbmep as mep
from jax import random, numpy as jnp, vmap, Array, lax
from jax.scipy.special import logsumexp

from onmep.agent.eig import EIG


def _n_eff(
    logw: jnp.ndarray,      # (S, ...)
    keepdims: bool = True
):
    w = jnp.exp(logw)                                           # (S, ...)
    n_eff = 1. / jnp.sum(w * w, axis=0, keepdims=keepdims)      # (1, ...)
    return n_eff, w


def _silverman_bandwidth(
    x: jnp.ndarray,         # (F, S, ...)
    logw: jnp.ndarray,      # (S, ...)
    eps: float = 1e-12,
):
    F = x.shape[0]
    n_eff, w = _n_eff(logw, keepdims=True)                      # (1, ...), (S, ...)

    mean = jnp.sum(
        w[None] * x,                                            # (F, S, ...)
        axis=1,
        keepdims=True,
    )                                                           # (F, 1, ...)

    var = jnp.sum(
        w[None] * (x - mean) ** 2,                              # (F, S, ...)
        axis=1,
        keepdims=True,
    )                                                           # (F, 1, ...)
    sd = jnp.sqrt(var)                                          # (F, 1, ...)

    scale = jnp.power(
        n_eff * (F + 2.) / 4.,                                  # (1, ...)
        -1. / (F + 4.),                                         # ()
    )                                                           # (1, ...)
    out = scale[None] * sd + eps                                # (F, 1, ...)
    return out[:, 0]                                            # (F, ...)


def _log_gaussian_kernel(
    z: jnp.ndarray,                                             # (F, S1, S2, ...)
):
    return -0.5 * jnp.sum(z * z, axis=0)                        # (S1, S2, ...)


def _log_gaussian_kernel_cov(
    x: jnp.ndarray,                                             # (F, S)
    logw: jnp.ndarray,                                          # (S,)
    idx: jnp.ndarray | None = None,                             # (num_inner,)
    ridge: float = 1e-6,
):
    F = x.shape[0]

    if idx is None:
        idx = slice(None)

    n_eff, w = _n_eff(logw, keepdims=False)                     # (), (S,)

    mean = jnp.sum(
        w[None] * x,                                            # (F, S)
        axis=1,
        keepdims=True,
    )                                                           # (F, 1)
    xc = x - mean                                               # (F, S)
    cov = jnp.einsum("s,fs,gs->fg", w, xc, xc)                  # (F, F)

    scale2 = jnp.power(
        n_eff * (F + 2.) / 4.,
        -2. / (F + 4.),
    )                                                           # ()
    cov_h = scale2 * cov + ridge * jnp.eye(F, dtype=x.dtype)    # (F, F)

    delta = x[..., None] - x[..., None, idx]                    # (F, S, num_inner)
    rhs = delta.reshape(F, -1)                                  # (F, S * num_inner)
    sol = jnp.linalg.solve(cov_h, rhs)                          # (F, S * num_inner)
    sol = sol.reshape(delta.shape)                              # (F, S, num_inner)

    log_kernel = -0.5 * jnp.sum(delta * sol, axis=0)            # (S, num_inner)
    return log_kernel


class FocusedEIG(EIG):
    def __init__(
        self,
        *args,
        focus_parameters: tuple[str, ...],
        kernel: str = "gaussian_cov",
        num_conditional: float | None = None,
        **kw,
    ):
        """
        Parameters
        ----------
        num_conditional
            Fraction of posterior samples used to estimate each inner
            conditional predictive density.

            If None, all posterior samples are used exactly.

            If in (0, 1], then ceil(num_conditional * S)
            inner samples are drawn with replacement from a proposal based on
            the kernel-conditioned particle weights, where S is the number of
            posterior samples. Importance weights correct for this proposal.
        """
        super().__init__(*args, **kw)

        implemented = ("gaussian", "gaussian_cov")
        if kernel not in implemented:
            raise NotImplementedError(
                f"kernel must be in {implemented}, got {kernel}"
            )

        self.name = "focused_eig"
        self.kernel = kernel
        self.num_conditional = num_conditional

        self._focus_params = focus_parameters
        self._focus_axes = None
        self._focus_unconstrained: jnp.ndarray | None = None
        return

    def _reset_focus(self):
        self._focus_axes = None
        self._focus_unconstrained = None
        return

    def _on_model_changed(self):
        super()._on_model_changed()
        self._reset_focus()
        _ = self.focus_axes
        return

    @property
    def focus_parameters(self):
        return self._focus_params

    @focus_parameters.setter
    def focus_parameters(self, val):
        self._focus_params = val
        self._reset_focus()
        return

    @property
    def focus_axes(self):
        if self._focus_axes is None:
            self._focus_axes = jnp.asarray([
                i
                for i, u in enumerate(self._args)
                if u in self.focus_parameters
            ])
        return self._focus_axes

    def _update_posterior(self, *args, extra_fields=(), **kw):
        self._focus_unconstrained = None
        extra_fields = tuple(f"z.{u}" for u in self.focus_parameters)
        return super()._update_posterior(*args, extra_fields=extra_fields, **kw)

    @property
    def focus_unconstrained(self) -> jnp.ndarray:
        if self._focus_unconstrained is None:
            z = self.posterior.array[len(self._args):]          # (F, S, K, R)
            self._focus_unconstrained = jnp.moveaxis(z, -1, 0)  # (R, F, S, K)

        return self._focus_unconstrained

    @staticmethod
    def _precompute_kernel(
        key,
        posterior: jnp.ndarray,                 # (F, S)
        logw: jnp.ndarray,                      # (S,), normalized
        kernel: str,
        num_inner: int | None = None,
    ):
        S = posterior.shape[1]

        if num_inner is None:
            idx = jnp.arange(S)
        else:
            idx = random.choice(key, S, shape=(num_inner,), replace=True)

        if kernel == "gaussian":
            bw = _silverman_bandwidth(posterior, logw=logw)     # (F,)
            z = (
                posterior[..., None]                            # (F, S, 1)
                - posterior[..., None, idx]                     # (F, 1, num_inner)
            ) / bw[:, None, None]                               # (F, 1, 1)
            logk = _log_gaussian_kernel(z)                      # (S, num_inner)

        elif kernel == "gaussian_cov":
            logk = _log_gaussian_kernel_cov(
                posterior, logw, idx=idx
            )                                                   # (S, num_inner)

        else:
            raise ValueError

        logw_cond = logw[idx][None, :] + logk                   # (S, num_inner)
        logw_cond -= logsumexp(
            logw_cond, axis=1, keepdims=True
        )                                                       # (S, num_inner)
        return logw_cond, idx                                   # (S, num_inner), (num_inner,)

    def eig_one(
        self,
        key: Array,
        x: float,                           # ()
        posterior: jnp.ndarray,             # (P, S)
        logw: jnp.ndarray,                  # (S,), normalized
        logw_cond: jnp.ndarray,             # (S, num_cond)
        idx_cond: jnp.ndarray,              # (num_cond,)
        num_outer: int | None = None,       # outer MC sample size
    ):
        S = posterior.shape[1]
        key, key_idx, key_y = random.split(key, 3)

        params = self._params_all(x, posterior)                 # (S,) each
        w = jnp.exp(logw)                                       # (S,)

        def body_one_draw(kkey, i):
            params_s = [param[i] for param in params]           # () each
            y_s = self._sample_y(kkey, *params_s)               # ()
            loglik_all = vmap(
                lambda *params_j: self._loglik(y_s, *params_j)
            )(*params)                                          # (S,)
            log_pred = logsumexp(loglik_all + logw, axis=0)     # ()

            params_cond = posterior[:, idx_cond]                # (P, num_cond)
            params_cond = params_cond.at[self.focus_axes].set(
                posterior[self.focus_axes, i][:, None]
            )                                                   # (P, num_cond)
            params_cond = self._params_all(x, params_cond)      # (num_cond) each

            loglik_cond = vmap(
                lambda *params_j: self._loglik(y_s, *params_j)
            )(*params_cond)                                     # (num_cond,)
            log_cond_pred = logsumexp(
                loglik_cond + logw_cond[i],
                axis=0,
            )                                                   # ()
            return log_cond_pred - log_pred                     # ()

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
        pos_rs = jnp.arange(num_rs)

        if logw.shape != posterior[0].shape:
            raise ValueError("logw and posterior sample size mismatch")

        keys_k = random.split(key, num_ks)                          # (num_ks,)

        logw_rs = logw[..., rs]                                     # (S, K, num_rs)
        S = posterior.shape[1]

        if self.num_conditional is None:
            num_cond = None
        else:
            num_cond = max(
                1,
                min(
                    S,
                    int(jnp.ceil(self.num_conditional * S)),
                ),
            )

        def one_k(k, key_k):
            focus_k = self.focus_unconstrained[..., k][rs]          # (num_rs, F, S)
            key_k, keys_cond = random.split(key_k)                  # () each
            keys_cond = random.split(keys_cond, num_rs)             # (num_rs,)
            logw_cond, idx_cond = vmap(
                self._precompute_kernel,
                in_axes=(0, 0, 0, None, None),
            )(
                keys_cond,                                          # (num_rs,)
                focus_k,                                            # (num_rs, F, S)
                logw_rs[:, k, :].T,                                 # (num_rs, S)
                self.kernel,
                num_cond,
            )                                                       # (num_rs, S, num_cond), (num_rs, num_cond)

            def one_x(x, key_x):
                def one_r(pos_r, r, key_r):
                    return self.eig_one(
                        key=key_r,
                        x=x,
                        posterior=posterior[..., k, r],
                        logw=logw[..., k, r],
                        logw_cond=logw_cond[pos_r],
                        idx_cond=idx_cond[pos_r],
                        num_outer=num_outer,
                    )
                keys_r = (
                    key_x[None]
                    if num_rs == 1
                    else random.split(key_x, num_rs)
                )                                               # (num_rs,)
                vals = vmap(one_r)(pos_rs, rs, keys_r)          # (num_rs,)
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

    def state_dict(
        self,
        include_posterior: bool = True,
    ) -> dict:
        d = super().state_dict(include_posterior=include_posterior)
        d["focused_eig"] = {
            "kernel": self.kernel,
            "num_conditional": self.num_conditional,
            "_focus_params": self._focus_params,
        }
        return d

    def load_state_dict(self, state: dict):
        super().load_state_dict(state)
        fd = state.get("focused_eig", {})
        self.kernel = fd.get("kernel", self.kernel)
        self.num_conditional = fd.get("num_conditional", self.num_conditional)
        self._focus_params = fd.get("_focus_params", self._focus_params)
        self._reset_focus()
        return self
