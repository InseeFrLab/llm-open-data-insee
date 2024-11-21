from typing import Any, TypeVar

T = TypeVar("T")


def compare_params(big_dict: dict[T, Any], small_dict: dict[T, Any]) -> list[T]:
    """
    Checks whether the parameters from `small_dict` are compatible from those in `big_dict`.

    Args:
    big_dict: a dict
    small_dict: a dict

    Returns:
    The keys of `small_dict` that are either missing from `big_dict`
    or that have different values in each input dict
    """
    return [key for key in small_dict if key not in big_dict or (big_dict[key] != small_dict[key])]
