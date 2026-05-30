from functools import partial

import hbmep as mep
from jax import random, numpy as jnp
from jax.scipy.special import gammaln

EPS = 1e-3
LOG2PI = jnp.log(2.0 * jnp.pi)


class LikelihoodMixin:
    @property
    def _model(self):
        return self._model_fn

    @_model.setter
    def _model(self, fn):
        self._model_fn = fn
        name = fn.__name__

        rl = (
            partial(mep.functional.rectified_logistic, eps=EPS),
            [mep.site.a, mep.site.b, mep.site.g, mep.site.h, mep.site.v],
        )

        gamma = (
            self._gamma_params_all,
            self._gamma_sample_y,
            self._gamma_loglik,
        )
        lognormal = (
            self._lognormal_params_all,
            self._lognormal_sample_y,
            self._lognormal_loglik,
        )

        if name in {"rl_gamma", "rl_rmt_gamma"}:
            self._fn, self._fn_args = rl
            self._params_all, self._sample_y, self._loglik = gamma

        elif name == "rl_lognormal":
            self._fn, self._fn_args = rl
            self._params_all, self._sample_y, self._loglik = lognormal

        elif name == "rl_delta_gamma":
            self._fn, self._fn_args = rl
            self._params_all, self._sample_y, self._loglik = gamma

        else:
            raise ValueError(f"Unknown model function assigned to _model: {name}")

    def _gamma_params_all(self, x: float, k: int, posterior: dict):
        mu = self._fn(x, *(posterior[u][:, k, :] for u in self._fn_args))  # (S, R)
        alpha, beta = mep.BaseModel.gamma_likelihood(
            mu,
            posterior[mep.site.c1][:, k, :],
            posterior[mep.site.c2][:, k, :]
        )   # (S, R)
        return alpha, beta

    @staticmethod
    def _gamma_sample_y(key, *params):
        """
        Uses:
            z ~ Gamma(site.alpha, 1) (in shape-rate parameterization)
            z / site.beta ~ Gamma(site.alpha, site.beta)
        """
        alpha, beta = params
        return random.gamma(key, a=alpha) / beta

    @staticmethod
    def _gamma_loglik(y, *params):
        """
        Joint Gamma(site.alpha, site.beta) log-likelihood across
        R muscles (conditionally independent). Inputs broadcast to
        (..., R) -> (...).
        """
        alpha, beta = params
        return jnp.sum(
            alpha * jnp.log(beta)
            - gammaln(alpha)
            + (alpha - 1.0) * jnp.log(y)
            - beta * y,
            axis=-1,
        )

    def _lognormal_params_all(self, x: float, k: int, posterior: dict):
        mu = self._fn(x, *(posterior[u][:, k, :] for u in self._fn_args))  # (S, R)
        loc = jnp.log(mu)                           # (S, R)
        scale = posterior[mep.site.c1][:, k, :]     # (S, R)
        return loc, scale

    @staticmethod
    def _lognormal_sample_y(key, *params):
        """
        Uses:
            eps ~ Normal(0, 1)
            exp(loc + scale * eps) ~ LogNormal(loc, scale)
        """
        loc, scale = params
        eps = random.normal(key, shape=loc.shape)   # (R,)
        return jnp.exp(loc + scale * eps)           # (R,)

    @staticmethod
    def _lognormal_loglik(y, *params):
        """
        Joint LogNormal(loc, scale) log-likelihood across
        R muscles (conditionally independent). Inputs broadcast to
        (..., R) -> (...).
        """
        loc, scale = params
        ly = jnp.log(y)
        z = (ly - loc) / scale
        return jnp.sum(
            -ly
            - jnp.log(scale)
            - 0.5 * LOG2PI
            - 0.5 * z * z,
            axis=-1,
        )
