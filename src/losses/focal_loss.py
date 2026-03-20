"""Focal loss for class-imbalanced classification.

FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

where p_t = softmax probability of the true class.

References:
  Lin et al. 2017, "Focal Loss for Dense Object Detection"
  gamma=2.0 down-weights well-classified examples (p_t > 0.5)
  alpha = sqrt-inverse class weights from dataset
"""
# ASSERT_CONVENTION: primary_metric=macro_f1, loss=focal_gamma2_sqrt_inverse_alpha

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss with per-class alpha weighting.

    Parameters
    ----------
    gamma : float
        Focusing parameter. gamma=0 reduces to weighted cross-entropy.
    alpha : Tensor or None
        Per-class weight tensor of shape (num_classes,). If None, uniform weighting.
    reduction : str
        'mean', 'sum', or 'none'.
    label_smoothing : float
        Label smoothing epsilon (0.0 = no smoothing).
    """

    def __init__(self, gamma=2.0, alpha=None, reduction="mean", label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits, targets):
        """Compute focal loss.

        Parameters
        ----------
        logits : Tensor (N, C)
            Raw model outputs (before softmax).
        targets : Tensor (N,)
            Ground truth class indices.

        Returns
        -------
        loss : Tensor
            Scalar (if reduction='mean' or 'sum') or (N,) if 'none'.
        """
        num_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)  # (N, C) -- numerically stable
        probs = torch.exp(log_probs)               # (N, C)

        # Apply label smoothing to targets
        if self.label_smoothing > 0:
            # Smooth targets: (1 - eps) for true class, eps/(C-1) for others
            eps = self.label_smoothing
            one_hot = F.one_hot(targets, num_classes).float()
            smooth_targets = one_hot * (1 - eps) + (1 - one_hot) * eps / (num_classes - 1)
            # Focal modulation on true-class probability
            p_t = (probs * one_hot).sum(dim=1)  # probability of true class
            focal_weight = (1 - p_t).pow(self.gamma)
            # Weighted cross-entropy with smoothed targets
            ce = -(smooth_targets * log_probs).sum(dim=1)
            loss = focal_weight * ce
        else:
            # Standard focal loss (no smoothing)
            # Gather log_prob and prob for the true class
            log_p_t = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (N,)
            p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)          # (N,)
            focal_weight = (1 - p_t).pow(self.gamma)
            loss = -focal_weight * log_p_t  # (N,)

        # Apply per-class alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha[targets]  # (N,)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def test_focal_loss():
    """Unit test: verify focal loss against analytic values.

    Test cases per plan:
    1. FL(p=0.9, gamma=2, alpha=0.25) = -0.25 * (0.1)^2 * log(0.9) = 0.000263
    2. FL(p=0.1, gamma=2, alpha=0.25) = -0.25 * (0.9)^2 * log(0.1) = 0.4669
    3. gamma=0 reduces to weighted cross-entropy
    """
    import math

    print("=== Focal Loss Unit Tests ===\n")

    # Test 1: FL(p=0.9, gamma=2, alpha=0.25)
    # For a two-class case, set logits so that softmax(class 0) ~ 0.9
    # softmax([x, 0]) = [e^x/(e^x+1), 1/(e^x+1)]
    # We want p=0.9 for class 0: e^x/(e^x+1) = 0.9 -> x = log(9) ~ 2.197
    p_target = 0.9
    logit = math.log(p_target / (1 - p_target))  # inverse sigmoid
    logits = torch.tensor([[logit, 0.0]])
    targets = torch.tensor([0])
    alpha = torch.tensor([0.25, 0.25])

    fl = FocalLoss(gamma=2.0, alpha=alpha, reduction="none", label_smoothing=0.0)
    loss_val = fl(logits, targets).item()
    expected = -0.25 * (1 - p_target)**2 * math.log(p_target)
    rel_error = abs(loss_val - expected) / abs(expected)
    print(f"Test 1: FL(p=0.9, gamma=2, alpha=0.25)")
    print(f"  Computed: {loss_val:.8f}")
    print(f"  Expected: {expected:.8f}")
    print(f"  Rel error: {rel_error:.2e}")
    assert rel_error < 1e-6, f"FAIL: rel_error={rel_error:.2e} > 1e-6"
    print(f"  PASS\n")

    # Test 2: FL(p=0.1, gamma=2, alpha=0.25)
    p_target = 0.1
    logit = math.log(p_target / (1 - p_target))
    logits = torch.tensor([[logit, 0.0]])
    targets = torch.tensor([0])

    loss_val = fl(logits, targets).item()
    expected = -0.25 * (1 - p_target)**2 * math.log(p_target)
    rel_error = abs(loss_val - expected) / abs(expected)
    print(f"Test 2: FL(p=0.1, gamma=2, alpha=0.25)")
    print(f"  Computed: {loss_val:.8f}")
    print(f"  Expected: {expected:.8f}")
    print(f"  Rel error: {rel_error:.2e}")
    assert rel_error < 1e-4, f"FAIL: rel_error={rel_error:.2e} > 1e-4"
    print(f"  PASS\n")

    # Test 3: gamma=0 should match weighted cross-entropy
    p_target = 0.7
    logit = math.log(p_target / (1 - p_target))
    logits = torch.tensor([[logit, 0.0]])
    targets = torch.tensor([0])

    fl_gamma0 = FocalLoss(gamma=0.0, alpha=alpha, reduction="none", label_smoothing=0.0)
    loss_focal = fl_gamma0(logits, targets).item()
    # Weighted CE: -alpha * log(p)
    expected_ce = -0.25 * math.log(p_target)
    rel_error = abs(loss_focal - expected_ce) / abs(expected_ce)
    print(f"Test 3: gamma=0 matches weighted CE (p=0.7)")
    print(f"  Focal(gamma=0): {loss_focal:.8f}")
    print(f"  Weighted CE:    {expected_ce:.8f}")
    print(f"  Rel error: {rel_error:.2e}")
    assert rel_error < 1e-6, f"FAIL: rel_error={rel_error:.2e} > 1e-6"
    print(f"  PASS\n")

    # Test 4: Gradient flows (basic check)
    logits_grad = torch.randn(4, 10, requires_grad=True)
    targets_grad = torch.randint(0, 10, (4,))
    alpha_10 = torch.ones(10)
    fl_grad = FocalLoss(gamma=2.0, alpha=alpha_10)
    loss = fl_grad(logits_grad, targets_grad)
    loss.backward()
    assert logits_grad.grad is not None, "FAIL: no gradient"
    assert not torch.isnan(logits_grad.grad).any(), "FAIL: NaN gradient"
    print(f"Test 4: Gradient flows without NaN")
    print(f"  PASS\n")

    print("=== All focal loss tests PASSED ===")
    return True


if __name__ == "__main__":
    test_focal_loss()
