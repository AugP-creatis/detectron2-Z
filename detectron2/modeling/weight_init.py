import torch.nn as nn
from collections.abc import Callable

def init_module(module: nn.Module, init_fn: Callable[[nn.Module], None]) -> None:   #inspirerd by nn.Module apply()
    """
    Allow to perform fvcore weight_init.py initializations also for a module which don't have a weight attribute.
    In that case, initialize its submodules which have a weight attribute.

    Args:
        module (torch.nn.Module): module to initialize.
    """
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
        init_fn(module)
    else:
        for submodule in module.children():
            if isinstance(submodule, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                init_fn(submodule)
