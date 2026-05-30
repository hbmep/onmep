import os
import logging
from dataclasses import dataclass, replace

import hbmep as mep
import numpy as np
import pandas as pd
from jax import random, Array, numpy as jnp
from jax.scipy.special import logsumexp
from scipy.stats import qmc
import numpyro as pyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC
from numpyro.infer.util import log_likelihood

from onmep import Observation, ObsArrays
from onmep.agent.generative import LikelihoodMixin, EPS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Action:
    x: float
    f: list[int]
    vals: jnp.ndarray | None = None

    def state_dict(self):
        vals = self.vals
        d = {
            "x": self.x,
            "f": list(self.f),
            "vals": None if vals is None else np.asarray(vals),
        }
        return d

    @classmethod
    def from_state(cls, state: dict):
        vals = state.get("vals")
        if vals is not None:
            vals = jnp.asarray(vals)
        obj = cls(x=state["x"], f=state["f"], vals=vals)
        return obj


@dataclass
class Proposal:
    action: Action
    phase: str
    shared: bool = False
    is_pending: bool = True

    def state_dict(self):
        d = {
            "action": self.action.state_dict(),
            "phase": self.phase,
            "shared": self.shared,
            "is_pending": self.is_pending,
        }
        return d

    @classmethod
    def from_state(cls, state):
        return cls(
            action=Action.from_state(state["action"]),
            phase=state["phase"],
            shared=state.get("shared", False),
            is_pending=state.get("is_pending", True),
        )


class BaseAgent(LikelihoodMixin, mep.BaseModel):
    def __init__(
        self,
        *,
        key: Array | None = None,
        toml_path: str | None = None,
        config: dict | None = None,
        infer: str = "mcmc",
        ess_threshold: float = 0.5,
        initial_capacity: int = 32,
        K: int = 1,     # possible values of features[0]
        N0: int = 8,    # num observations at 0
        N1: int = 3,    # num random observations for warmup/kickstart
        grid_lo: float = 0.,
        grid_hi: float = 100.,
        grid_n: int = 101,
        grid_max: float = 100.,     # effective grid maximum
        update_every: int = 1,      # set to 0 to disable updates (e.g. for uniform)
        use_acquisition: bool = True,
        share_zero: bool = False,
        share_warm: bool = False,
    ) -> None:
        mep.BaseModel.__init__(self, key=key, toml_path=toml_path, config=config)

        if self.num_features > 1:
            raise NotImplementedError(f"Not implemented for {self.num_features} features.")

        # RNG: key: mep.BaseModel and _key: BaseAgent
        self.key, self._key = random.split(self.key)
        self.name: str = "base_agent"

        implemented = ("mcmc", "sequential", "svi")
        if infer not in implemented:
            raise NotImplementedError(f"infer must be in {implemented}, got {infer}")

        self.infer = infer
        self.ess_threshold = ess_threshold

        self.data = ObsArrays(
            D=self.num_features,
            R=self.num_response,
            initial_capacity=initial_capacity
        )

        self.K = K
        self.N0 = N0
        self.N1 = N1
        self.share_zero = share_zero
        self.share_warm = share_warm

        self.grid_n = grid_n
        self.grid_x = jnp.linspace(grid_lo, grid_hi, grid_n, dtype=self.data.dtype_x)
        self.grid_f = jnp.arange(self.K, dtype=self.data.dtype_f)
        self.grid_max = grid_max

        self.update_every = update_every
        self.use_acquisition = use_acquisition

        self._perm_x: np.ndarray | None = None
        self._perm_x_i: int = 0
        self._last_proposal: Proposal | None = None

        self._mcmc = {"kernel": None, "state": None}
        self.posterior: dict | None = None
        self.logw = None
        return

    @property
    def _num_zero(self) -> int:
        return self.N0 * self.K

    @property
    def _num_warm(self) -> int:
        return self.N1 * self.K

    @property
    def _num_start(self) -> int:
        return self._num_zero + self._num_warm

    @property
    def _in_zero_phase(self) -> bool:
        return self.data.n < self._num_zero

    @property
    def _in_warm_phase(self) -> bool:
        return self._num_zero <= self.data.n < self._num_start

    @property
    def _in_postwarm_phase(self) -> bool:
        return self.data.n >= self._num_start

    def _zero_propose(self) -> Action:
        return Action(x=0., f=[self.data.n % self.K])

    @staticmethod
    def _init_perm_x(*, key, N0, grid_x, grid_max):
        """
        Initialize a scrambled Halton ordering of grid_x indices for uniform
        proposals, so that prefix subsequences are roughly equispaced.

        If grid_x starts at 0 and N0 > 0, index 0 is excluded so that
        the random policy never proposes zero intensity.
        """
        seed = int(random.randint(key, (), 0, 2 ** 31 - 1))
        sampler = qmc.Halton(d=1, scramble=True, seed=seed)
        x = sampler.random(n=grid_x.shape[0])[:, 0]
        perm = np.argsort(x)

        grid_x = np.asarray(grid_x)
        valid = grid_x <= grid_max

        if grid_x.min() == 0. and N0 > 0:
            valid = valid & (grid_x != 0.)

        perm = perm[valid[perm]]
        return perm

    def _uniform_propose(self) -> Action:
        if (self._perm_x is None) or (self._perm_x_i >= self._perm_x.shape[0]):
            self._key, sub = random.split(self._key)
            perm = self._init_perm_x(
                key=sub,
                N0=self.N0,
                grid_x=self.grid_x,
                grid_max=self.grid_max,
            )
            self._perm_x = perm
            self._perm_x_i = 0

        idx = self._perm_x[self._perm_x_i]
        x = float(self.grid_x[idx])
        k = self.data.n % self.K

        if k == self.K - 1:
            self._perm_x_i += 1

        return Action(x=x, f=[k])

    def _compute_acquisition(
        self,
        *,
        feature0: int | None = None,
        response: int | None = None,
    ) -> jnp.ndarray:
        """
        Compute acquisition values over the (K, N) grid.

        Returns
        -------
        vals : shape (K, N)
            Values for each feature k and grid point x, where
            K = self.K and N = self.grid_x.shape[0].

            vals[k, i] corresponds to the value at feature k and grid_x[i].
            The proposal is chosen by taking the argmax over vals,
            i.e., higher values are more favorable.
        """
        raise NotImplementedError

    def _propose(
        self,
        *,
        feature0: int | None = None,
        response: int | None = None
    ) -> Action:
        vals = self._compute_acquisition(feature0=feature0, response=response)  # (K, N)

        mask_x = self.grid_x <= self.grid_max                   # (N,)
        masked = jnp.where(mask_x[None, :], vals, -jnp.inf)     # (K, N)

        flat = jnp.argmax(masked)
        N = self.grid_x.shape[0]

        ix_best = int(flat % N)
        x_best = float(self.grid_x[ix_best])
        k_best = int(flat // N)

        return Action(x=x_best, f=[k_best], vals=vals)

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
            action = self._propose(feature0=feature0, response=response)

        else:
            phase = "postwarm"
            shared = False
            action = self._uniform_propose()

        self._last_proposal = Proposal(
            action=action, phase=phase, shared=shared, is_pending=True
        )
        return action

    @staticmethod
    def _require_data(data: ObsArrays, msg: str):
        if not data.n:
            raise ValueError(msg)
        return

    @staticmethod
    def _model_kwargs(data: ObsArrays, only_valid=True):
        if only_valid:
            return dict(
                intensity=data.X,
                features=data.F,
                response=data.Y,
            )
        return dict(
            intensity=data._X,
            features=data._F,
            response=data._Y,
            num_used=data.n,
        )

    def _update_mcmc(
        self,
        *,
        data: ObsArrays,
        extra_fields=(),
        init_params=None,
        **kw
    ) -> tuple[dict, jnp.ndarray]:
        self._require_data(data, "No observations in buffer, collect data before run_mcmc().")
        model_kw = self._model_kwargs(data)

        if self._mcmc["kernel"] is None:
            self._mcmc["kernel"] = NUTS(self._model, **self.nuts_kw)
        mcmc = MCMC(self._mcmc["kernel"], **self.mcmc_kw)

        self.key, sub = random.split(self.key)
        mcmc.run(sub, **model_kw, extra_fields=extra_fields, init_params=init_params, **kw)
        self._mcmc["state"] = mcmc

        posterior = mcmc.get_samples()
        posterior.pop(mep.site.mu, None)
        posterior.pop(mep.site.obs, None)

        S = next(iter(posterior.values())).shape[0]
        logw = jnp.zeros(S)
        return posterior, logw

    def _update_sequential(self, *, data: ObsArrays, **kw) -> tuple[dict, jnp.ndarray]:
        if self.posterior is None:
            return self._update_mcmc(data=data, **kw)

        model_kw = dict(
            intensity=data.X[-1:],
            features=data.F[-1:],
            response=data.Y[-1:],
        )

        ll = log_likelihood(self._model, self.posterior, **model_kw, **kw)
        u = ll[mep.site.obs]
        loglik = u.sum(tuple(range(1, u.ndim))).reshape(-1)

        logw = self.logw + loglik
        logw = logw - logsumexp(logw)

        w = jnp.exp(logw)
        ess = 1.0 / jnp.sum(w * w)

        S = next(iter(self.posterior.values())).shape[0]
        threshold = (
            self.ess_threshold * S
            if self.ess_threshold <= 1.0
            else self.ess_threshold
        )
        if ess < threshold:
            return self._update_mcmc(data=data, **kw)

        return self.posterior, logw

    def trace(
        self,
        df: pd.DataFrame | None = None,
        key: Array | None = None,
        **kw
    ):
        row = [0.0] + [0] * self.num_features + [1.0] * self.num_response
        df = pd.DataFrame([row], columns=self.regressors + self.response)

        num_features = [1]
        num_response = self.num_response
        use_mixture = self.use_mixture

        trace = super().trace(
            df=df,
            key=key,
            num_features=num_features,
            num_response=num_response,
            use_mixture=use_mixture
        )
        return trace

    def append_obs(self, obs: Observation):
        shared = self._last_proposal.shared

        if not shared:
            self.data.append_obs(obs)
            return

        for k in range(self.K):
            obs_k = replace(obs, features=obs.features.at[0].set(k))
            self.data.append_obs(obs_k)
        return

    def _empty_like_posterior(self, posterior: dict[str, jnp.ndarray]):
        out = {}
        for name, v in posterior.items():
            S = v.shape[0]
            out[name] = jnp.full(
                (S, self.K, self.num_response),
                jnp.nan,
                dtype=v.dtype,
            )
        return out

    def _update_posterior(
        self,
        *,
        data: ObsArrays | None = None,
        feature0: int | None = None,
        response: int | None = None,
    ):
        if data is None:
            data = self.data

        num_features = [self.K]
        num_response = self.num_response
        use_mixture = self.use_mixture

        if feature0 is not None:
            data = data.at_feature0(feature0, remap=True)
            num_features = [1]

        if response is not None:
            data = data.at_response(response)
            num_response = 1

        update_kw = dict(
            data=data,
            num_features=num_features,
            num_response=num_response,
            use_mixture=use_mixture
        )

        match self.infer:
            case "mcmc":
                posterior_new, logw_new = self._update_mcmc(**update_kw)
            case "sequential":
                if feature0 is not None or response is not None:
                    raise ValueError("sequential update does not support sliced updates")
                posterior_new, logw_new = self._update_sequential(**update_kw)
            case _:
                raise NotImplementedError

        if feature0 is None and response is None:
            self.posterior = posterior_new
            self.logw = logw_new
            return

        fs = slice(feature0, feature0 + 1) if feature0 is not None else slice(None)
        rs = slice(response, response + 1) if response is not None else slice(None)

        if self.posterior is None:
            self.posterior = self._empty_like_posterior(posterior_new)

        for name, v in posterior_new.items():
            if name in {mep.site.outlier_prob}:
                self.posterior[name] = v
            else:
                self.posterior[name] = self.posterior[name].at[:, fs, rs].set(v)

        self.logw = logw_new
        return

    def should_stop(self):
        return False

    def _should_update(
        self,
        *,
        feature0: int | None = None,
        response: int | None = None,
    ) -> bool:
        if not self._in_postwarm_phase:
            return False
        n_after_warm = self.data.n - self._num_start
        should_update = (
            self.update_every > 0
            and (
                self.posterior is None
                or n_after_warm % self.update_every == 0
            )
        )
        return should_update

    def update(
        self,
        obs: Observation,
        *,
        feature0: int | None = None,
        response: int | None = None,
    ):
        self.append_obs(obs)
        if self._last_proposal is not None:
            self._last_proposal.is_pending = False

        should_update = self._should_update(feature0=feature0, response=response)
        if should_update:
            self._update_posterior(feature0=feature0, response=response)

        return self.should_stop()

    def _on_response_changed(self, old, new):
        if getattr(self, "data", None) is not None:
            self.data.resize_R(len(new))
        return

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop("_mcmc", None)
        return d

    def _reset_mcmc(self):
        self._mcmc = {"kernel": None, "state": None}
        return

    def __setstate__(self, d):
        self.__dict__.update(d)
        self._reset_mcmc()
        return

    def state_dict(self, include_posterior: bool = True) -> dict:
        key_data = random.key_data(self._key)

        perm_x = None
        if self._perm_x is not None:
            perm_x = np.asarray(self._perm_x, dtype=int).tolist()

        last_proposal = None
        if self._last_proposal is not None:
            last_proposal = self._last_proposal.state_dict()

        out = {
            "base_model": super().state_dict(),
            "_key_data": key_data.tolist(),
            "_key_dtype": str(key_data.dtype.name),
            "name": self.name,

            "infer": self.infer,
            "ess_threshold": self.ess_threshold,

            "data": self.data.state_dict(),
            "K": self.K,
            "N0": self.N0,
            "N1": self.N1,
            "share_zero": self.share_zero,
            "share_warm": self.share_warm,

            "grid_lo": np.asarray(self.grid_x).min(),
            "grid_hi": np.asarray(self.grid_x).max(),
            "grid_n": self.grid_n,
            "grid_max": self.grid_max,

            "update_every": self.update_every,
            "use_acquisition": self.use_acquisition,

            "_perm_x": perm_x,
            "_perm_x_i": self._perm_x_i,
            "_last_proposal": last_proposal,
        }

        if include_posterior:
            posterior = None
            if self.posterior is not None:
                remove_keys = (mep.site.mu, mep.site.obs)
                posterior = {
                    k: np.asarray(v)
                    for k, v in self.posterior.items()
                    if k not in remove_keys
                }
            out["posterior"] = posterior
            out["logw"] = None if self.logw is None else np.asarray(self.logw)

        return out

    def load_state_dict(self, state: dict):
        super().load_state_dict(state.get("base_model", {}))

        key_data = state.get("_key_data", [0, 0])
        key_dtype = jnp.dtype(state.get("_key_dtype", jnp.uint32))
        self._key = random.wrap_key_data(jnp.array(key_data, dtype=key_dtype))

        self.name = state.get("name", self.name)

        self.infer = state.get("infer", self.infer)
        self.ess_threshold = state.get("ess_threshold", self.ess_threshold)

        self.data = ObsArrays.from_state(state["data"])
        self.K = state.get("K", self.K)
        self.N0 = state.get("N0", self.N0)
        self.N1 = state.get("N1", self.N1)
        self.share_zero = state.get("share_zero", self.share_zero)
        self.share_warm = state.get("share_warm", self.share_warm)

        grid_lo = state.get("grid_lo", 0.)
        grid_hi = state.get("grid_hi", 100.)
        self.grid_n = state.get("grid_n", self.grid_n)
        self.grid_x = jnp.linspace(grid_lo, grid_hi, self.grid_n, dtype=self.data.dtype_x)
        self.grid_f = jnp.arange(self.K, dtype=self.data.dtype_f)
        self.grid_max = state.get("grid_max", self.grid_max)

        self.update_every = state.get("update_every", self.update_every)
        self.use_acquisition = state.get("use_acquisition", self.use_acquisition)

        perm_x = state.get("_perm_x", self._perm_x)
        perm_x_i = state.get("_perm_x_i", 0)
        if perm_x is None:
            self._perm_x = None
            self._perm_x_i = 0
        else:
            arr = np.asarray(perm_x, dtype=int)
            self._perm_x = arr
            self._perm_x_i = min(perm_x_i, arr.shape[0])

        last_proposal = state.get("_last_proposal", self._last_proposal)
        if last_proposal is not None:
            last_proposal = Proposal.from_state(last_proposal)
        self._last_proposal = last_proposal

        posterior = state.get("posterior", self.posterior)
        if posterior is not None:
            self.posterior = {u: jnp.asarray(v) for u, v in posterior.items()}

        logw = state.get("logw", None)
        self.logw = jnp.asarray(logw) if logw is not None else None

        self._reset_mcmc()
        return self

    def _base_plotter(
        self,
        predictive_hdi_prob: float = .95,
        num_points=100
    ):
        if self.posterior is None:
            raise ValueError("No posterior available. Run update first.")

        df = self.data.to_df(**self.variables)
        posterior = {u: np.array(v) for u, v in self.posterior.items()}
        posterior.pop(mep.site.mu, None)
        posterior.pop(mep.site.obs, None)
        prediction_df = self.make_prediction_dataset(df, num_points=num_points)
        ppd = self.predict(
            df=prediction_df,
            posterior=posterior,
            num_response=self.num_response,
            num_features=[self.K],
            use_mixture=self.use_mixture,
        )

        figures = mep.plot(
            df=df,
            **self.variables,
            prediction_df=prediction_df,
            predictive=ppd,
            posterior=posterior,
            predictive_hdi_prob=predictive_hdi_prob,
        )
        figures = [fig for fig, _ in figures]
        return figures

    def _plot_curves(
        self,
        output_path: str | None = None,
        predictive_hdi_prob: float = .95,
        num_points=100
    ):
        if output_path:
            ...
        elif self.build_dir:
            output_path = os.path.join(self.build_dir, "curves.pdf")
        else:
            return

        figures = self._base_plotter(
            predictive_hdi_prob=predictive_hdi_prob,
            num_points=num_points
        )
        mep.make_pdf(figures, output_path=output_path)
        return

    @staticmethod
    def _sample_priors(
        *,
        num_features,
        num_response,
        include_c2: bool = True,
        truncate_g: bool = False,
        **kw
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
        num_response,
        num_used,
        use_mixture,
        mu,
        lik,
        lik_params,
        **kw
    ):
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

    def rl_gamma(self, intensity, features, response=None, num_used=None, **kw):
        include_c2 = True
        lik = "gamma"
        fn_params, lik_params = self._sample_priors(include_c2=include_c2, **kw)

        mu = mep.functional.rectified_logistic(
            intensity,
            **{u: v[*features.T] for u, v in fn_params.items()},
            eps=EPS
        )

        self._observe(
            intensity=intensity,
            features=features,
            response=response,
            num_used=num_used,
            mu=mu,
            lik=lik,
            lik_params=lik_params,
            **kw
        )

    def rl_lognormal(self, intensity, features, response=None, num_used=None, **kw):
        include_c2 = False
        lik = "lognormal"
        fn_params, lik_params = self._sample_priors(include_c2=include_c2, **kw)

        mu = mep.functional.rectified_logistic(
            intensity,
            **{u: v[*features.T] for u, v in fn_params.items()},
            eps=EPS
        )

        self._observe(
            intensity=intensity,
            features=features,
            response=response,
            num_used=num_used,
            mu=mu,
            lik=lik,
            lik_params=lik_params,
            **kw
        )

    def rl_rmt_gamma(self, intensity, features, response=None, num_used=None, **kw):
        include_c2 = True
        lik = "gamma"
        fn_params, lik_params = self._sample_priors(include_c2=include_c2, truncate_g=True, **kw)

        mu = mep.functional.rectified_logistic(
            intensity,
            **{u: v[*features.T] for u, v in fn_params.items()},
            eps=EPS
        )

        self._observe(
            intensity=intensity,
            features=features,
            response=response,
            num_used=num_used,
            mu=mu,
            lik=lik,
            lik_params=lik_params,
            **kw
        )

    @staticmethod
    def _sample_delta_priors(
        *,
        num_features,
        num_response,
        target_parameter,
        **kw
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

    def rl_delta_gamma(self, intensity, features, response=None, num_used=None, **kw):
        lik = "gamma"
        fn_params, lik_params = self._sample_delta_priors(
            target_parameter=self.target_parameter, **kw
        )

        mu = mep.functional.rectified_logistic(
            intensity,
            **{u: v[*features.T] for u, v in fn_params.items()},
            eps=EPS
        )

        self._observe(
            intensity=intensity,
            features=features,
            response=response,
            num_used=num_used,
            mu=mu,
            lik=lik,
            lik_params=lik_params,
            **kw
        )
