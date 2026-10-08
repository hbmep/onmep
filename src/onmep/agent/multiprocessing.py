import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import atexit

from jax import random, numpy as jnp
from numpyro.infer import NUTS, MCMC

from onmep.agent.dtypes import Posterior

_EXECUTORS: dict[tuple[str, int], ProcessPoolExecutor] = {}

_WORKER_KERNEL = None
_WORKER_MODEL = None


def _get_executor(
    owner_id: str,
    max_workers: int,
) -> ProcessPoolExecutor:
    key = (owner_id, max_workers)
    executor = _EXECUTORS.get(key)
    if executor is None:
        context = mp.get_context("spawn")
        executor = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=context,
        )
        _EXECUTORS[key] = executor
    return executor


def _shutdown_executors():
    for executor in _EXECUTORS.values():
        executor.shutdown(wait=True)
    _EXECUTORS.clear()
    return

atexit.register(_shutdown_executors)


def _mcmc_one(args):
    global _WORKER_KERNEL, _WORKER_MODEL
    agent, data, key_data, fi, ri, model_kw, extra_fields = args
    key = random.wrap_key_data(jnp.asarray(key_data, dtype=jnp.uint32))

    data_i = data.at_feature0(fi, remap=True)
    data_i = data_i.at_response(ri)

    if _WORKER_KERNEL is None:
        _WORKER_MODEL = partial(agent._model, **model_kw, num_features=[1], num_response=1)
        _WORKER_KERNEL = NUTS(_WORKER_MODEL, **agent.nuts_kw)

    mcmc = MCMC(_WORKER_KERNEL, **agent.mcmc_kw)
    mcmc.run(key, **agent._model_kwargs(data_i), extra_fields=extra_fields)

    samples = mcmc.get_samples()
    named_params = list(agent._args)

    if extra_fields:
        samples.update(mcmc.get_extra_fields())
        named_params += list(extra_fields)

    posterior = Posterior.from_dict(posterior=samples, named_params=named_params)
    return fi, ri, posterior
