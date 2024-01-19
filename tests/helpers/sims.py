from caustics.namespace_dict import NestedNamespaceDict
from caustics.sims.state_dict import _sanitize


def extract_tensors(params, include_params=False):
    # Extract the "static" and "dynamic" parameters
    param_dicts = list(params.values())

    # Extract the "static" and "dynamic" parameters
    # to a single merged dictionary
    final_dict = NestedNamespaceDict()
    for pdict in param_dicts:
        for k, v in pdict.items():
            if k not in final_dict:
                final_dict[k] = v
            else:
                final_dict[k] = {**final_dict[k], **v}

    # flatten function only exists for NestedNamespaceDict
    all_params = final_dict.flatten()

    tensors_dict = _sanitize({k: v.value for k, v in all_params.items()})
    if include_params:
        return tensors_dict, all_params
    return tensors_dict


def isEquals(a, b):
    # Go through each key and values
    # change empty torch to be None
    # since we can't directly compare
    # empty torch
    truthy = []
    for k, v in a.items():
        if k not in b:
            return False
        kv = b[k]
        if (v.nelement() == 0) or (kv.nelement() == 0):
            v = None
            kv = None
        truthy.append(v == kv)

    return all(truthy)
