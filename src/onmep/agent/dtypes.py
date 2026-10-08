from dataclasses import dataclass
from typing import Any, Callable, TypedDict

import numpy as np
from jax import numpy as jnp
from jax.scipy.special import logsumexp
from numpyro.infer import NUTS, MCMC


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

    def assert_equal(
        self,
        other: "Action",
        *,
        rtol: float = 1e-7,
        atol: float = 0.0,
    ):
        if not isinstance(other, Action):
            raise TypeError(f"Expected Action, got {type(other)}")

        np.testing.assert_allclose(self.x, other.x, rtol=rtol, atol=atol)
        assert self.f == other.f

        if self.vals is None or other.vals is None:
            assert self.vals is other.vals
        else:
            np.testing.assert_allclose(
                np.asarray(self.vals),
                np.asarray(other.vals),
                rtol=rtol,
                atol=atol,
            )
        return


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

    def assert_equal(
        self,
        other: "Proposal",
        *,
        rtol: float = 1e-7,
        atol: float = 0.0,
    ):
        if not isinstance(other, Proposal):
            raise TypeError(f"Expected Proposal, got {type(other)}")

        self.action.assert_equal(other.action, rtol=rtol, atol=atol)

        assert self.phase == other.phase
        assert self.shared == other.shared
        assert self.is_pending == other.is_pending
        return


class MCMCState(TypedDict):
    state: MCMC | None
    kernel: NUTS | None
    model: Callable[..., Any] | None


class Posterior:
    """
    Named posterior with shape
        array : (P, S, K, R)
        logw  : (S, K, R)
    where
        P : number of named parameters
        S : posterior samples
        K : features
        R : responses
    """
    def __init__(
        self,
        array: jnp.ndarray,                         # (P, S, K, R)
        named_params: tuple[str, ...] | list[str],
        logw: jnp.ndarray | None = None,            # (S, K, R)
    ):
        self._array = jnp.asarray(array)
        self.named_params = tuple(named_params)
        self._mapping = {name: i for i, name in enumerate(self.named_params)}

        if self._array.ndim != 4:
            raise ValueError(
                f"Posterior array must have shape (P, S, K, R), "
                f"got {self._array.shape}"
            )

        if self._array.shape[0] != len(self.named_params):
            raise ValueError(
                f"P={self._array.shape[0]} but got "
                f"{len(self.named_params)} named_params"
            )
        
        if logw is None:
            logw = jnp.zeros(self._array.shape[1:])
        else:
            logw = jnp.asarray(logw)
            if logw.shape != self._array.shape[1:]:
                raise ValueError(
                    f"logw must have shape {self._array.shape[1:]}, "
                    f"got {logw.shape}"
                )

        self._logw = logw - logsumexp(logw, axis=0, keepdims=True)
        self._w = None
        return

    @classmethod
    def from_dict(
        cls,
        posterior: dict[str, jnp.ndarray],
        named_params: tuple[str, ...] | list[str],
        logw: jnp.ndarray | None = None,
    ):
        missing = [u for u in named_params if u not in posterior]
        if missing:
            raise KeyError(f"Missing posterior sites: {missing}")
        array = jnp.stack([posterior[u] for u in named_params], axis=0)     # (P, S, K, R)
        return cls(array=array, named_params=named_params, logw=logw)

    @property
    def array(self):
        return self._array

    @array.setter
    def array(self, value):
        value = jnp.asarray(value)
        if value.shape != self._array.shape:
            raise ValueError(
                f"Expected shape {self._array.shape}, got {value.shape}"
            )
        self._array = value
        return

    @property
    def logw(self):
        return self._logw

    @logw.setter
    def logw(self, value):
        value = jnp.asarray(value)
        if value.shape != self._logw.shape:
            raise ValueError(
                f"Expected shape {self._logw.shape}, got {value.shape}"
            )
        self._logw = value - logsumexp(value, axis=0, keepdims=True)
        self._w = None
        return

    @property
    def w(self):
        if self._w is None:
            self._w = jnp.exp(self._logw)
        return self._w

    @property
    def shape(self):
        return self._array.shape

    @property
    def P(self):
        return self._array.shape[0]

    @property
    def S(self):
        return self._array.shape[1]

    @property
    def K(self):
        return self._array.shape[2]

    @property
    def R(self):
        return self._array.shape[3]

    def index(self, name: str) -> int:
        return self._mapping[name]

    def _resolve_key(self, key):
        if isinstance(key, str):
            return self._mapping[key]

        if isinstance(key, (list, tuple)) and all(
            isinstance(k, str) for k in key
        ):
            return jnp.asarray([self._mapping[k] for k in key])

        return key

    def __getitem__(self, key):
        return self._array[self._resolve_key(key)]

    def keys(self):
        return self.named_params

    def values(self):
        for name in self.named_params:
            yield self[name]

    def items(self):
        for name in self.named_params:
            yield name, self[name]

    def __contains__(self, key: str):
        return key in self._mapping

    def __iter__(self):
        return iter(self.named_params)

    def __len__(self):
        return len(self.named_params)

    def assert_equal(
        self,
        other: "Posterior",
        *,
        rtol: float = 1e-7,
        atol: float = 0.0,
    ):
        if not isinstance(other, Posterior):
            raise TypeError(f"Expected Posterior, got {type(other)}")

        assert tuple(self.named_params) == tuple(other.named_params)

        np.testing.assert_allclose(
            np.asarray(self.array),
            np.asarray(other.array),
            rtol=rtol,
            atol=atol,
        )

        np.testing.assert_allclose(
            np.asarray(self.logw),
            np.asarray(other.logw),
            rtol=rtol,
            atol=atol,
        )
        return

    def state_dict(self):
        return {
            "array": np.asarray(self.array),
            "logw": np.asarray(self.logw),
            "named_params": list(self.named_params),
        }

    @classmethod
    def from_state(cls, state: dict):
        return cls(
            array=jnp.asarray(state["array"]),
            logw=jnp.asarray(state["logw"]),
            named_params=tuple(state["named_params"]),
        )

    def __setitem__(self, key, value):
        if isinstance(key, str):
            key = self._mapping[key]

        value = jnp.asarray(value)

        try:
            self._array = self._array.at[key].set(value)
        except ValueError as exc:
            raise ValueError(
                f"Cannot assign value with shape {value.shape} "
                f"to posterior slice with shape {self._array[key].shape}"
            ) from exc

    def as_dict(self) -> dict[str, jnp.ndarray]:
        return dict(self.items())
