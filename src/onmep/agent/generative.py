from functools import partial

import hbmep as mep
from jax import numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist

from onmep.agent.likelihood import EPS


class GenerativeMixin:
    @staticmethod
    def _sample_priors(
        *,
        num_features,
        num_response,
        include_c2: bool = True,
        truncate_g: bool = False,
    ):
        fn_params, lik_params = {}, {}
        with pyro.plate(mep.site.num_response, num_response):
            with pyro.plate(mep.site.num_features[0], num_features[0]):
                fn_params[mep.site.a] = pyro.sample(
                    mep.site.a, dist.TruncatedNormal(50., 50., low=0.)
                )
                fn_params[mep.site.b] = pyro.sample(mep.site.b, dist.HalfNormal(1.))

                if truncate_g:
                    fn_params[mep.site.g] = pyro.sample(
                        mep.site.g,
                        dist.TruncatedNormal(0., 0.1, low=0., high=0.045)
                    )
                else:
                    fn_params[mep.site.g] = pyro.sample(mep.site.g, dist.HalfNormal(.1))

                fn_params[mep.site.h] = pyro.sample(mep.site.h, dist.HalfNormal(5.))
                fn_params[mep.site.v] = pyro.sample(mep.site.v, dist.HalfNormal(5.))

                lik_params[mep.site.c1] = pyro.sample(mep.site.c1, dist.HalfNormal(5.))
                if include_c2:
                    lik_params[mep.site.c2] = pyro.sample(mep.site.c2, dist.HalfNormal(.5))

        return fn_params, lik_params

    @staticmethod
    def _observe(
        *,
        intensity,
        features,
        response,
        num_used,
        num_response,
        use_mixture,
        fn,
        fn_params,
        lik,
        lik_params,
    ):
        mu = fn(x=intensity, **{u: v[*features.T] for u, v in fn_params.items()})

        num_data = intensity.shape[0]

        mask_obs = True
        if num_used is not None:
            mask_obs = (jnp.arange(num_data) < num_used)[:, None]
        if response is not None:
            mask_obs = mask_obs & jnp.invert(jnp.isnan(response))

        if use_mixture:
            if response is None:
                q = 0.
            else:
                q = pyro.sample(mep.site.outlier_prob, dist.Uniform(0., 0.01))

        with pyro.handlers.mask(mask=mask_obs):
            with pyro.plate(mep.site.num_response, num_response):
                with pyro.plate(mep.site.num_data, num_data):
                    pyro.deterministic(mep.site.mu, mu)
                    if lik == "gamma":
                        alpha, beta = mep.BaseModel.gamma_likelihood(
                            mu,
                            lik_params[mep.site.c1][*features.T],
                            lik_params[mep.site.c2][*features.T]
                        )
                        fg = dist.Gamma(concentration=alpha, rate=beta)
                    elif lik == "lognormal":
                        loc = jnp.log(mu)
                        scale = lik_params[mep.site.c1][*features.T]
                        fg = dist.LogNormal(loc=loc, scale=scale)
                    else:
                        raise ValueError

                    if use_mixture:
                        mixing_distribution = dist.Categorical(
                            probs=jnp.stack([1 - q, q], axis=-1)
                        )
                        component_distributions = [
                            fg, dist.HalfNormal(scale=10.)
                        ]
                        dist_ = dist.MixtureGeneral(
                            mixing_distribution=mixing_distribution,
                            component_distributions=component_distributions,
                        )
                    else:
                        dist_ = fg

                    pyro.sample(mep.site.obs, dist_, obs=response)

    def gamma(
        self,
        intensity,
        features,
        response=None,
        num_used=None,
        num_features=None,
        num_response=None,
        use_mixture=None,
    ):
        include_c2 = True
        truncate_g = False
        fn = partial(mep.functional.rectified_logistic, eps=EPS)
        lik = "gamma"

        fn_params, lik_params = self._sample_priors(
            num_features=num_features,
            num_response=num_response,
            include_c2=include_c2,
            truncate_g=truncate_g,
        )

        self._observe(
            intensity=intensity,
            features=features,
            response=response,
            num_used=num_used,
            num_response=num_response,
            use_mixture=use_mixture,
            fn=fn,
            fn_params=fn_params,
            lik=lik,
            lik_params=lik_params,
        )

    def lognormal(
        self,
        intensity,
        features,
        response=None,
        num_used=None,
        num_features=None,
        num_response=None,
        use_mixture=None,
    ):
        include_c2 = False
        truncate_g = False
        fn = partial(mep.functional.rectified_logistic, eps=EPS)
        lik = "lognormal"

        fn_params, lik_params = self._sample_priors(
            num_features=num_features,
            num_response=num_response,
            include_c2=include_c2,
            truncate_g=truncate_g,
        )

        self._observe(
            intensity=intensity,
            features=features,
            response=response,
            num_used=num_used,
            num_response=num_response,
            use_mixture=use_mixture,
            fn=fn,
            fn_params=fn_params,
            lik=lik,
            lik_params=lik_params,
        )

    def rmt_gamma(
        self,
        intensity,
        features,
        response=None,
        num_used=None,
        num_features=None,
        num_response=None,
        use_mixture=None,
    ):
        include_c2 = True
        truncate_g = True
        fn = partial(mep.functional.rectified_logistic, eps=EPS)
        lik = "gamma"

        fn_params, lik_params = self._sample_priors(
            num_features=num_features,
            num_response=num_response,
            include_c2=include_c2,
            truncate_g=truncate_g,
        )

        self._observe(
            intensity=intensity,
            features=features,
            response=response,
            num_used=num_used,
            num_response=num_response,
            use_mixture=use_mixture,
            fn=fn,
            fn_params=fn_params,
            lik=lik,
            lik_params=lik_params,
        )

    @staticmethod
    def _sample_delta_priors(
        *,
        num_features,
        num_response,
        target_parameter,
    ):
        num_fixed = 1
        num_delta = num_features[0] - 1

        fn_params, lik_params = {}, {}
        with pyro.plate(mep.site.num_response, num_response):
            with pyro.plate("num_fixed", num_fixed):
                a_fixed = pyro.sample("a_fixed", dist.TruncatedNormal(50., 50., low=0.))

            with pyro.plate("num_delta", num_delta):
                a_delta = pyro.sample("a_delta", dist.Normal(0., 50.))

            if target_parameter == mep.site.a:
                ...

            elif target_parameter == "delta":
                pyro.deterministic("delta", jnp.concatenate([a_fixed, a_delta], axis=0))

            else:
                raise ValueError(f"Invalid target_parameter {target_parameter}")

            fn_params[mep.site.a] = pyro.deterministic(
                mep.site.a, jnp.concatenate([a_fixed, a_fixed + a_delta], axis=0)
            )

            with pyro.plate(mep.site.num_features[0], num_features[0]):
                fn_params[mep.site.b] = pyro.sample(mep.site.b, dist.HalfNormal(1.))
                fn_params[mep.site.g] = pyro.sample(mep.site.g, dist.HalfNormal(.1))

                fn_params[mep.site.h] = pyro.sample(mep.site.h, dist.HalfNormal(5.))
                fn_params[mep.site.v] = pyro.sample(mep.site.v, dist.HalfNormal(5.))

                lik_params[mep.site.c1] = pyro.sample(mep.site.c1, dist.HalfNormal(5.))
                lik_params[mep.site.c2] = pyro.sample(mep.site.c2, dist.HalfNormal(.5))

        return fn_params, lik_params

    def delta_gamma(
        self,
        intensity,
        features,
        response=None,
        num_used=None,
        num_features=None,
        num_response=None,
        use_mixture=None,
    ):
        fn = partial(mep.functional.rectified_logistic, eps=EPS)
        lik = "gamma"

        fn_params, lik_params = self._sample_delta_priors(
            num_features=num_features,
            num_response=num_response,
            target_parameter=self.target_parameter,
        )

        self._observe(
            intensity=intensity,
            features=features,
            response=response,
            num_used=num_used,
            num_response=num_response,
            use_mixture=use_mixture,
            fn=fn,
            fn_params=fn_params,
            lik=lik,
            lik_params=lik_params,
        )
