"""
Useful functions.
"""

from torch.nn import Module


def count_parameters(model: Module, requires_grad: bool = False) -> int:
    """Counts the parameters in a torch.nn.Module object."""
    return sum(p.numel() for p in model.parameters() if (not requires_grad or p.requires_grad))


def pformat_log(log: list[dict[str, float]]) -> dict[str, list[float]]:
    """Pretty formats the log for plotting."""
    new_log = {k: [] for k in log[0].keys()}
    for l in log:
        for k, v in l.items():
            new_log[k].append(v)
    return new_log
