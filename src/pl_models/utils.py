from torch.nn.modules.loss import _Loss
from torch import nn

__all__ = ["get_loss_function"]


def get_loss_function(loss_name: str) -> _Loss:
    if (loss_function := getattr(nn, loss_name)) is not None:
        return loss_function()
    else:
        raise ValueError(f"`{loss_name}` is not a supported loss function!")
