import time
import logging
from dataclasses import dataclass, field

import numpy as np
import jax.numpy as jnp

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Observation:
    intensity: float                # ()
    features: jnp.ndarray           # (D,)
    response: jnp.ndarray           # (R,)
    id: int = 0                     # ()
    t_sec: float = field(default_factory=lambda: time.time())
    info: dict = field(default_factory=dict)

    def __repr__(self):
        f_str = ", ".join([f"{x}" for x in self.features])
        r_str = ", ".join([f"{x:.2f}" for x in self.response])
        return (
            f"Obs(int={self.intensity:.2f}, feat=[{f_str}], "
            f"resp=[{r_str}])"
        )

    def to_string(self):
        return self.__repr__()

    def state_dict(self):
        return {
            "intensity": float(self.intensity),
            "features": np.asarray(self.features),
            "response": np.asarray(self.response),
            "id": int(self.id),
            "t_sec": float(self.t_sec),
            "info": dict(self.info),
        }

    @classmethod
    def from_state(cls, state: dict):
        return cls(
            intensity=state["intensity"],
            features=jnp.asarray(state["features"]),
            response=jnp.asarray(state["response"]),
            id=state.get("id", 0),
            t_sec=state.get("t_sec", time.time()),
            info=dict(state.get("info", {})),
        )


class ObsArrays:
    """
    Growable (X, F, Y) buffer with shapes:
      X: (N, 1)         intensity
      F: (N, D)         features
      Y: (N, R)         response
    """
    def __init__(
        self,
        *,
        D: int | None = None,
        R: int | None = None,
        initial_capacity: int = 32,
    ):
        self.D = D
        self.R = R

        self.dtype_x = jnp.float64
        self.dtype_f = jnp.int32
        self.dtype_y = jnp.float64
        self.dtype_o = jnp.int32

        self._cap = initial_capacity
        self._n = 0     # num_data

        self._X = jnp.empty((self._cap, 1), dtype=self.dtype_x)
        self._F = jnp.empty(
            (self._cap, 0 if self.D is None else self.D),
            dtype=self.dtype_f
        )
        self._Y = jnp.empty(
            (self._cap, 0 if self.R is None else self.R),
            dtype=self.dtype_y
        )
        self._O = jnp.empty((self._cap,), dtype=self.dtype_o)
        return

    def __len__(self):
        return self._n

    @property
    def n(self):    # num_data
        return self._n

    @property
    def capacity(self):
        return self._cap

    @property
    def shape(self):
        return (self._n, self.D, self.R)

    @property
    def capacity_shape(self):
        return (self._cap, self.D, self.R)

    @property
    def X(self):
        return self._X[:self._n, :]

    @property
    def F(self):
        return self._F[:self._n, :]

    @property
    def Y(self):
        return self._Y[:self._n, :]

    @property
    def O(self):
        return self._O[:self._n]

    def _grow(self, new_cap: int):
        new_cap = max(new_cap, self._cap * 2)
        logger.info(f"Capacity changed from {self._cap} to {new_cap}")

        X_new = jnp.empty((new_cap, 1), dtype=self.dtype_x)
        X_new = X_new.at[:self._n, :].set(self._X[:self._n, :])

        if self.D is None:
            F_new = jnp.empty((new_cap, 0), dtype=self.dtype_f)
        else:
            F_new = jnp.empty((new_cap, self.D), dtype=self.dtype_f)
            if self._n:
                F_new = F_new.at[:self._n, :].set(self._F[:self._n, :])

        if self.R is None:
            Y_new = jnp.empty((new_cap, 0), dtype=self.dtype_y)
        else:
            Y_new = jnp.empty((new_cap, self.R), dtype=self.dtype_y)
            if self._n:
                Y_new = Y_new.at[:self._n, :].set(self._Y[:self._n, :])

        O_new = jnp.empty((new_cap,), dtype=self.dtype_o)
        O_new = O_new.at[:self._n].set(self._O[:self._n])

        self._X = X_new
        self._F = F_new
        self._Y = Y_new
        self._O = O_new
        self._cap = new_cap
        return

    def append(
        self,
        *,
        intensity: float,
        features: jnp.ndarray,
        response: jnp.ndarray,
        id: int = 0
    ):
        x = jnp.asarray(intensity, dtype=self.dtype_x)
        f = jnp.asarray(features, dtype=self.dtype_f).reshape(-1)
        r = jnp.asarray(response, dtype=self.dtype_y).reshape(-1)
        o = jnp.asarray(id, dtype=self.dtype_o)

        if self.D is None:
            self.D = f.size
            self._F = jnp.empty((self._cap, self.D), dtype=self.dtype_f)
        elif f.size != self.D:
            raise ValueError(f"Features length {f.size} != fixed D {self.D}")

        if self.R is None:
            self.R = r.size
            self._Y = jnp.empty((self._cap, self.R), dtype=self.dtype_y)
        elif r.size != self.R:
            raise ValueError(f"Response length {r.size} != fixed R {self.R}")

        if self._n >= self._cap:
            self._grow(self._cap * 2)

        self._X = self._X.at[self._n, 0].set(x)
        self._F = self._F.at[self._n, :].set(f)
        self._Y = self._Y.at[self._n, :].set(r)
        self._O = self._O.at[self._n].set(o)
        self._n += 1
        return

    def append_obs(self, obs: Observation):
        self.append(
            intensity=obs.intensity,
            features=obs.features,
            response=obs.response,
            id=obs.id,
        )

    def resize_R(self, R_new: int):
        R_new = R_new
        if self.R == R_new:
            return

        Y_new = jnp.empty((self._cap, R_new), dtype=self.dtype_y)
        if self.R is not None:
            copy_cols = min(self.R, R_new)
            if copy_cols and self._n:
                Y_new = Y_new.at[:self._n, :copy_cols].set(self._Y[:self._n, :copy_cols])

        self._Y = Y_new
        self.R = R_new
        return

    # TODO
    def resize_F(self, F_new: int):
        raise NotImplementedError

    def to_df(
        self,
        *,
        intensity: str,
        features: list[str],
        response: list[str],
        id: str = "id",
        **kw
    ):
        import pandas as pd
        if self.R is None:
            return pd.DataFrame(columns=[intensity, *features, *response, id])
        if len(features) != self.D:
            raise ValueError(f"feature columns({len(features)}) != D({self.D})")
        if len(response) != self.R:
            raise ValueError(f"response columns({len(response)}) != R({self.R})")

        X_host = np.asarray(self.X)
        F_host = np.asarray(self.F)
        Y_host = np.asarray(self.Y)
        O_host = np.asarray(self.O)

        data = {intensity: X_host.ravel()}
        for j, col in enumerate(features):
            data[col] = F_host[:, j]
        for j, col in enumerate(response):
            data[col] = Y_host[:, j]
        data[id] = O_host

        return pd.DataFrame(data)

    def state_dict(self) -> dict:
        return {
            "D": self.D,
            "R": self.R,
            "_cap": self._cap,
            "_n": self._n,
            "X": np.asarray(self.X),
            "F": np.asarray(self.F),
            "Y": np.asarray(self.Y),
            "O": np.asarray(self.O),
        }

    @classmethod
    def from_state(cls, state: dict):
        cap = state["_cap"]
        n = state["_n"]
        D = state["D"]
        R = state["R"]

        obj = cls(D=D, R=R, initial_capacity=cap)
        obj._n = n

        # Restore arrays
        x_src = jnp.asarray(state["X"], dtype=obj.dtype_x)
        f_src = jnp.asarray(state["F"], dtype=obj.dtype_f)
        y_src = jnp.asarray(state["Y"], dtype=obj.dtype_y)
        if "O" in state:
            o_src = jnp.asarray(state["O"], dtype=obj.dtype_o)
        else:
            o_src = jnp.zeros((n,), dtype=obj.dtype_o)

        assert x_src.shape == (n, 1)
        f_cols = 0 if D is None else D
        assert f_src.shape == (n, f_cols)
        y_cols = 0 if R is None else R
        assert y_src.shape == (n, y_cols)
        assert o_src.shape == (n,)

        obj._X = obj._X.at[:n, :].set(x_src)
        if f_cols:
            obj._F = obj._F.at[:n, :].set(f_src)
        else:
            obj._F = jnp.empty((obj._cap, 0), dtype=obj.dtype_f)
        if y_cols:
            obj._Y = obj._Y.at[:n, :].set(y_src)
        else:
            obj._Y = jnp.empty((obj._cap, 0), dtype=obj.dtype_y)
        obj._O = obj._O.at[:n].set(o_src)

        return obj

    def at_feature0(self, k: int, *, remap: bool = False):
        mask = np.asarray(self.F[:, 0]) == k
        n = mask.sum()
        out = ObsArrays(
            D=self.D,
            R=self.R,
            initial_capacity=max(n, 1),
        )
        out._n = n
        out._X = out._X.at[:n, :].set(self.X[mask])
        out._F = out._F.at[:n, :].set(self.F[mask])
        out._Y = out._Y.at[:n, :].set(self.Y[mask])
        out._O = out._O.at[:n].set(self.O[mask])
        if remap:
            out._F = out._F.at[:n, 0].set(0)
        return out

    def at_response(self, r: int):
        out = ObsArrays(
            D=self.D,
            R=1,
            initial_capacity=max(self._n, 1),
        )
        out._n = self._n
        out._X = out._X.at[:self._n, :].set(self.X)
        out._F = out._F.at[:self._n, :].set(self.F)
        out._Y = out._Y.at[:self._n, :].set(self.Y[:, r:r + 1])
        out._O = out._O.at[:self._n].set(self.O)
        return out

    def at_index(self, i: int) -> Observation:
        return Observation(
            intensity=float(self._X[i, 0]),
            features=self._F[i],
            response=self._Y[i],
            id=int(self._O[i]),
        )
