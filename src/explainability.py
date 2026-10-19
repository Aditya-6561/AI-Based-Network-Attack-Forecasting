import numpy as np
import torch

from src.state_builder import FEATURE_ORDER

# Feature names are sourced from the state builder so the two can never drift apart.
FEATURE_NAMES = list(FEATURE_ORDER)


class IntegratedGradientsExplainer:
    """
    Integrated Gradients attribution over the observed history window.

    Attributes the K-step forecast risk back onto the (W x D) input history, then
    aggregates over time to give a per-feature contribution. Attribution is taken
    against the risk LOGIT rather than the sigmoid output: once the sigmoid
    saturates its gradient vanishes and every attribution collapses to ~0.
    """

    def __init__(self, model, steps: int = 32, feature_names=None):
        self.model = model
        self.steps = steps
        self.feature_names = list(feature_names) if feature_names else FEATURE_NAMES

    def _target(self, x, horizon=None):
        _, risk_logits, _ = self.model(x)
        if horizon is None:
            return risk_logits.max()
        return risk_logits[:, horizon].sum()

    def _integrate(self, input_seq, horizon, baseline):
        diff = input_seq - baseline
        total_grad = torch.zeros_like(input_seq)

        # Riemann midpoint rule over the straight-line path baseline -> input.
        for i in range(self.steps):
            alpha = (i + 0.5) / self.steps
            point = (baseline + alpha * diff).clone().detach().requires_grad_(True)
            self.model.zero_grad(set_to_none=True)
            self._target(point, horizon).backward()
            total_grad += point.grad.detach()

        return ((total_grad / self.steps) * diff).detach().cpu().numpy()

    def attribute(self, input_seq: torch.Tensor, horizon: int = None,
                  baseline: torch.Tensor = None):
        """
        input_seq: (1, W, D) already-normalised history.
        horizon:   which forecast step to explain; None = peak risk over the rollout.
        Returns a list of (feature_name, signed_attribution) sorted by |attribution|.
        """
        self.model.eval()
        if baseline is None:
            # Zero in normalised space is the training-set median window, i.e. a
            # "typical benign network" reference rather than an empty network.
            baseline = torch.zeros_like(input_seq)

        attributions = self._integrate(input_seq, horizon, baseline)  # (1, W, D)

        # Sum (not mean) over the history axis: IG is additive over input elements,
        # so summing preserves the completeness property per feature.
        feat_scores = attributions[0].sum(axis=0)

        results = [
            (self.feature_names[i] if i < len(self.feature_names) else "feature_%d" % i,
             float(feat_scores[i]))
            for i in range(feat_scores.shape[0])
        ]
        results.sort(key=lambda kv: abs(kv[1]), reverse=True)
        return results

    def temporal_attribution(self, input_seq: torch.Tensor, horizon: int = None):
        """Per-timestep total attribution -- shows *when* in the history risk accrued."""
        self.model.eval()
        baseline = torch.zeros_like(input_seq)
        attributions = self._integrate(input_seq, horizon, baseline)[0]
        return attributions.sum(axis=1)  # (W,)
