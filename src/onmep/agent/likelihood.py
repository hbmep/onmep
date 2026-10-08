from functools import partial

import hbmep as mep
from jax import random, numpy as jnp
from jax.scipy.special import gammaln

EPS = 1e-3
LOG2PI = jnp.log(2.0 * jnp.pi)


class LikelihoodMixin:
    def _on_model_changed(self):
        return

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

        l5 = (
            partial(mep.functional.logistic5),
            [mep.site.a, mep.site.b, mep.site.g, mep.site.h, mep.site.v],
        )

        gamma = (
            self._gamma_params_all,
            self._gamma_sample_y,
            self._gamma_loglik,
            [mep.site.c1, mep.site.c2],
        )
        lognormal = (
            self._lognormal_params_all,
            self._lognormal_sample_y,
            self._lognormal_loglik,
            [mep.site.c1],
        )

        if name in {"rl_gamma", "rmt_gamma"}:
            self._fn, self._fn_args = rl
            self._params_all, self._sample_y, self._loglik, self._lik_args = gamma

        elif name in {"rl_lognormal"}:
            self._fn, self._fn_args = rl
            self._params_all, self._sample_y, self._loglik, self._lik_args = lognormal

        elif name == "delta_gamma":
            self._fn, self._fn_args = rl
            self._params_all, self._sample_y, self._loglik, self._lik_args = gamma

        elif name in {"l5_gamma"}:
            self._fn, self._fn_args = l5
            self._params_all, self._sample_y, self._loglik, self._lik_args = gamma

        elif name.startswith("experimental"):
            ...

        else:
            raise ValueError(f"Unknown model function assigned to _model: {name}")
        
        self._args = self._fn_args + self._lik_args
        self._on_model_changed()
        return

    def _gamma_params_all(self, x: float, posterior: jnp.ndarray):
        """
        x: ()
        posterior: (P, ...)
        """
        mu = self._fn(x, *posterior[:len(self._fn_args)])
        alpha, beta = mep.BaseModel.gamma_likelihood(mu, *posterior[-len(self._lik_args):])
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
        alpha, beta = params
        return (
            alpha * jnp.log(beta)
            - gammaln(alpha)
            + (alpha - 1.0) * jnp.log(y)
            - beta * y
        )

    def _lognormal_params_all(self, x: float, posterior: jnp.ndarray):
        mu = self._fn(x, *posterior[:len(self._fn_args)])
        loc = jnp.log(mu)
        scale = posterior[-1]
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
        return (
            -ly
            - jnp.log(scale)
            - 0.5 * LOG2PI
            - 0.5 * z * z
        )
