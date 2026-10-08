import os
import logging

import hbmep as mep
import numpy as np
import jax.numpy as jnp
from jax.scipy.stats import norm

from onmep import ObsArrays
from onmep.agent.base_agent import BaseAgent

logger = logging.getLogger(__name__)

EPS = 1e-10


def _compute_ll_mle1(intensity, response_bin, mask_obs, a_grid, eps=EPS):
    """
    intensity: (N, 1)
    response_bin: (N, R)
    mask_obs: (N, R)
    a_grid: (A,)
    """
    b_grid = jnp.clip(0.07 * a_grid, 1e-3, None)    # (A,)

    prob = norm.cdf(
        x=intensity[:, None],
        loc=a_grid[None, :, None],
        scale=b_grid[None, :, None],
    )   # (N, A, 1)

    prob = jnp.clip(prob, eps, 1. - eps)

    ll = (
        response_bin[:, None] * jnp.log(prob)
        + (1. - response_bin[:, None]) * jnp.log(1. - prob)
    )   # (N, A, R)

    ll = jnp.where(mask_obs[:, None], ll, 0.)
    ll = jnp.sum(ll, axis=0)    # (A, R)
    return ll


def _compute_ll_mle2(intensity, response_bin, mask_obs, a_grid, b_grid, eps=EPS):
    """
    intensity: (N, 1)
    response_bin: (N, R)
    mask_obs: (N, R)
    a_grid: (A,)
    b_grid: (B,)
    """
    prob = norm.cdf(
        intensity[:, None, None],
        loc=a_grid[None, :, None, None],
        scale=b_grid[None, None, :, None],
    )  # (N, A, B, 1)

    prob = jnp.clip(prob, eps, 1. - eps)

    ll = (
        response_bin[:, None, None] * jnp.log(prob)
        + (1. - response_bin[:, None, None]) * jnp.log(1. - prob)
    )   # (N, A, B, R)

    ll = jnp.where(mask_obs[:, None, None], ll, 0.)
    ll = jnp.sum(ll, axis=0)    # (A, B, R)
    return ll


def run_pest(
    intensity: np.ndarray | jnp.ndarray,
    features: np.ndarray | jnp.ndarray,
    response: np.ndarray | jnp.ndarray,
    num_used: int,
    a_grid: np.ndarray | jnp.ndarray,
    b_grid: np.ndarray | jnp.ndarray,
    method: str = "mle1",
    mep_threshold: float = 0.05,
    eps: float = EPS,
):
    """
    intensity: (N, 1)
    features: (N, 1) (currently unused)
    response: (N, R)
    num_used: scalar n <= N
    a_grid: (A,)
    b_grid: (B,)
    """
    num_data = intensity.shape[0]   # = N (>= num_used)

    mask_obs = (jnp.arange(num_data) < num_used)[:, None]   # (N, 1)
    mask_obs = mask_obs & jnp.invert(jnp.isnan(response))   # (N, R)

    # Binary response
    response_bin = (response >= mep_threshold).astype(response.dtype)   # (N, R)

    if method == "mle1":
        ll = _compute_ll_mle1(
            intensity,
            response_bin,
            mask_obs,
            a_grid,
            eps=eps,
        )  # (A, R)

        max_ll = jnp.max(ll, axis=0)
        mask = ll >= max_ll[None, :] - eps

        a_best = (
            jnp.sum(
                jnp.where(mask, a_grid[:, None], 0.),
                axis=0,
            )
            / jnp.sum(mask, axis=0)
        )                                                   # (R,)
        b_best = jnp.clip(0.07 * a_best, 1e-3, None)        # (R,)

    elif method == "mle2":
        ll = _compute_ll_mle2(
            intensity,
            response_bin,
            mask_obs,
            a_grid,
            b_grid,
            eps=eps,
        )  # (A, B, R)

        max_ll = jnp.max(ll, axis=(0, 1))              # (R,)
        mask_eps = ll >= max_ll[None, None, :] - eps   # (A, B, R)
        mask_eps_count = jnp.sum(mask_eps, axis=(0, 1))

        a_grid_3d = jnp.broadcast_to(a_grid[:, None, None], ll.shape)   # (A, B, R)
        b_grid_3d = jnp.broadcast_to(b_grid[None, :, None], ll.shape)   # (A, B, R)

        a_best = (
            jnp.sum(jnp.where(mask_eps, a_grid_3d, 0.), axis=(0, 1))
            / mask_eps_count
        )   # (R,)

        b_best = (
            jnp.sum(jnp.where(mask_eps, b_grid_3d, 0.), axis=(0, 1))
            / mask_eps_count
        )   # (R,)

    else:
        raise NotImplementedError(f"{method} not implemented.")

    return a_best, b_best


class PEST(BaseAgent):
    def __init__(
        self,
        *args,
        method: str = "mle1",
        mep_threshold: float = 0.05,
        a_grid_n: int = 1001,
        b_grid_n: int = 501,
        **kw,
    ):
        super().__init__(*args, **kw)

        if self.K != 1:
            raise NotImplementedError(f"Not implemented for multiple features {self.K}")

        implemented = ("mle1", "mle2")
        if method not in implemented:
            raise NotImplementedError(f"method must be in {implemented}, got {method}")

        self.name = "pest"
        self.method = method
        self.mep_threshold = mep_threshold

        self.a_grid_n = a_grid_n
        self.b_grid_n = b_grid_n
        self._init_a_grid()
        self._init_b_grid()

        self._pest = {mep.site.a: None, mep.site.b: None}
        return

    @property
    def grid_max_eff(self):
        return self._grid_max

    @grid_max_eff.setter
    def grid_max_eff(self, value):
        self._grid_max = value
        if hasattr(self, "a_grid_n"):
            self._init_a_grid()
        return

    def _init_a_grid(self):
        grid_min = self.grid_x.min()
        grid_max = min(self.grid_max_eff, self.grid_x.max())
        self.a_grid = jnp.linspace(grid_min, grid_max, self.a_grid_n)
        return

    def _init_b_grid(self):
        b_grid_lo = 0.1
        b_grid_hi = 50.0
        log_min = jnp.log10(b_grid_lo)
        log_max = jnp.log10(b_grid_hi)
        log_centers = jnp.linspace(log_min, log_max, self.b_grid_n)
        self.b_grid = 10 ** log_centers
        return

    def _choose_response(self, response: int | None) -> int:
        if response is not None:
            return response
        if self.num_response == 1:
            return 0
        num_postwarm = self.data.n - self._num_start
        return num_postwarm % self.num_response

    def _update_pest(self, data: ObsArrays):
        self._require_data(data, "No observations in buffer, collect data before running PEST.")
        model_kw = self._model_kwargs(data, use_capacity=True)
        a_best, b_best = run_pest(
            **model_kw,
            a_grid=self.a_grid,
            b_grid=self.b_grid,
            method=self.method,
            mep_threshold=self.mep_threshold,
            eps=EPS,
        )
        return a_best, b_best

    def _compute_acquisition(
        self,
        *,
        feature0: int | None = None,
        response: int | None = None,
    ):
        data_pest = self.data
        a_best, b_best = self._update_pest(data=data_pest)
        self._pest[mep.site.a] = a_best
        self._pest[mep.site.b] = b_best

        if self._pest[mep.site.a] is None:
            raise ValueError("No pest available for acquisition.")

        if feature0 not in (None, 0):
            raise NotImplementedError("PEST currently supports only feature0=0.")

        response_idx = self._choose_response(response)
        target = self._pest[mep.site.a][response_idx]

        vals = -jnp.abs(self.grid_x[None, :] - target)
        return vals

    def state_dict(self, include_posterior: bool = True) -> dict:
        d = super().state_dict(include_posterior=include_posterior)
        d["pest"] = {
            "method": self.method,
            "mep_threshold": self.mep_threshold,
            "a_grid_n": self.a_grid_n,
            "b_grid_n": self.b_grid_n,
            mep.site.a: (
                None
                if self._pest[mep.site.a] is None
                else np.asarray(self._pest[mep.site.a])
            ),
            mep.site.b: (
                None
                if self._pest[mep.site.b] is None
                else np.asarray(self._pest[mep.site.b])
            ),
        }
        return d

    def load_state_dict(self, state: dict):
        super().load_state_dict(state)

        pd = state.get("pest", {})
        self.method = pd.get("method", self.method)
        self.mep_threshold = pd.get("mep_threshold", self.mep_threshold)

        self.a_grid_n = pd.get("a_grid_n", self.a_grid_n)
        self.b_grid_n = pd.get("b_grid_n", self.b_grid_n)
        self._init_a_grid()
        self._init_b_grid()

        a = pd.get(mep.site.a, None)
        b = pd.get(mep.site.b, None)

        self._pest = {
            mep.site.a: jnp.asarray(a) if a is not None else None,
            mep.site.b: jnp.asarray(b) if b is not None else None,
        }
        return self

    def _pest_plotter(self):
        import matplotlib.pyplot as plt
        import seaborn as sns

        if self._pest[mep.site.a] is None or self._pest[mep.site.b] is None:
            return []

        colors = sns.color_palette(
            palette="rainbow",
            n_colors=self.num_response,
        )

        a = np.asarray(self._pest[mep.site.a])
        b = np.asarray(self._pest[mep.site.b])

        intensity = np.asarray(self.data.X)
        response = np.asarray(self.data.Y)
        response_bin = (response >= self.mep_threshold).astype(response.dtype)

        x = np.linspace(float(self.grid_x.min()), float(self.grid_x.max()), 100)
        prob = norm.cdf(
            x=x[:, None],
            loc=a[None, :],
            scale=b[None, :],
        )

        nr, nc = 1, self.num_response
        fig, axes = plt.subplots(
            nr,
            nc,
            figsize=(5 * nc, 2.5 * nr),
            constrained_layout=True,
            squeeze=False,
            sharex=True,
            sharey=True,
        )

        for response_idx in range(self.num_response):
            ax = axes[0, response_idx]
            sns.scatterplot(
                x=intensity[..., 0],
                y=response_bin[..., response_idx],
                color=colors[response_idx],
                ax=ax,
            )
            sns.lineplot(
                x=x,
                y=np.asarray(prob)[..., response_idx],
                ax=ax,
                color="k",
                alpha=0.4,
            )
            ax.set_title(self.response[response_idx])
            ax.set_xlabel(self.intensity)
            ax.set_ylabel("response >= threshold")

        return [fig]

    def _plot_curves(self, output_path: str | None = None):
        if output_path:
            ...
        elif self.build_dir:
            output_path = os.path.join(self.build_dir, "curves.pdf")
        else:
            return

        figures = []
        if self.posterior is not None:
            figures += self._base_plotter()

        figures += self._pest_plotter()

        if not figures:
            return

        mep.make_pdf(figures=figures, output_path=output_path)
        return
