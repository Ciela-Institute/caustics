import functools
import inspect

import torch


def _args_to_kwargs(func, args):
    signature = inspect.signature(func)
    return {k: v for k, v in zip(signature.parameters.keys(), args)}


def _sync_device(kwargs):
    # Get all the tensor device types
    tensor_device_types = {
        k: v.device.type for k, v in kwargs.items() if isinstance(v, torch.Tensor)
    }

    # Get the set of devices
    device_set = set([t for t in tensor_device_types.values() if t != "cpu"])
    if device_set:
        # Get the first device
        device = device_set.pop()
        new_tensors = {
            k: kwargs[k].to(device)
            for k, v in tensor_device_types.items()
            if v == "cpu"
        }
        kwargs.update(new_tensors)
    return kwargs


def sync_device(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        kwargs.update(_args_to_kwargs(func, args))
        kwargs = _sync_device(kwargs)

        # Do something before
        value = func(**kwargs)
        # Do something after
        return value

    return wrapper_decorator
