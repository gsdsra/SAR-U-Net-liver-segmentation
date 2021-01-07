"""

Tversky loss
"""

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss


class TverskyLoss(_Loss):
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, eps: float = 1e-7):
        super(TverskyLoss, self).__init__()
        """
        Computes the Tversky loss [1].
        Args:
            alpha: controls the penalty for false positives.假阳性
            beta: controls the penalty for false negatives.假阴性
            eps: added to the denominator for numerical stability.
        Notes:
            alpha = beta = 0.5 => dice coeff
            alpha = beta = 1 => tanimoto coeff
            alpha + beta = 1 => F beta coeff
        References:
            [1]: https://arxiv.org/abs/1706.05721
        """
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, logits: Tensor, true: Tensor) -> Tensor:
        """
        Args:
            logits: a tensor of shape [B, C, H, W]. Corresponds to
                    the raw output or logits of the model.
            true: a tensor of shape [B, H, W] or [B, 1, H, W].

        Returns:
            tversky_loss: the Tversky loss.
        """
        num_classes = logits.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(logits, dim=1)

        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        fps = torch.sum(probas * (1 - true_1_hot), dims)
        fns = torch.sum((1 - probas) * true_1_hot, dims)

        num = intersection
        denom = intersection + (self.alpha * fps) + (self.beta * fns)
        tversky_loss = (num / (denom + self.eps)).mean()

        return 1 - tversky_loss


__all__ = ["TverskyLoss"]
