import torch
import torch.nn as nn
from typing import Tuple

NUM_STAGES = 7  # matches configs/attack_mapping.yaml stage_names (0..6)


class StateEncoder(nn.Module):
    """
    Encoder: maps state history S_{t-W:t} (B, W, D) to a latent vector z_t (B, L).
    """

    def __init__(self, input_dim: int = 42, hidden_dim: int = 64,
                 latent_dim: int = 32, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.fc_latent = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (h_n, c_n) = self.lstm(x)
        final_h = h_n[-1]
        return torch.tanh(self.fc_latent(final_h))


class LatentTransitionCell(nn.Module):
    """
    Transition engine T: z_{t+k-1} -> z_{t+k}. This is the component that makes the
    system a world model rather than a classifier: it advances the latent state
    without ever observing future traffic.
    """

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Residual latent update keeps the rollout stable over K steps.
        return z + self.net(z)


class StateDecoder(nn.Module):
    """Decoder: z_{t+k} -> predicted observable state S_{t+k} (B, D)."""

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 64, output_dim: int = 42):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class RiskHead(nn.Module):
    """Risk head: z_{t+k} -> logit of P(attack at t+k). Returns a LOGIT."""

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class StageHead(nn.Module):
    """
    ATT&CK stage head: z_{t+k} -> logits over MITRE ATT&CK stage indices.

    Stage is predicted from the *forecast* latent state, so the tactic label the
    dashboard shows is a model output about the future, not a lookup of the
    current ground-truth label.
    """

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 32,
                 num_stages: int = NUM_STAGES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_stages),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class TemporalWorldModel(nn.Module):
    """
    Temporal latent state-transition world model.

    Encodes an observed history into z_t, then rolls the latent state forward K
    steps through T(.) autoregressively. Ground-truth future states are never fed
    back during rollout -- at every step k the model conditions only on its own
    prediction z_{t+k-1}.

    forward() returns raw logits for risk and stage; use predict() for probabilities.
    """

    def __init__(self, input_dim: int = 42, hidden_dim: int = 64, latent_dim: int = 32,
                 forecast_steps: int = 6, num_stages: int = NUM_STAGES, dropout: float = 0.1):
        super().__init__()
        self.forecast_steps = forecast_steps
        self.num_stages = num_stages
        self.encoder = StateEncoder(input_dim, hidden_dim, latent_dim, dropout=dropout)
        self.transition = LatentTransitionCell(latent_dim, hidden_dim, dropout=dropout)
        self.decoder = StateDecoder(latent_dim, hidden_dim, input_dim)
        self.risk_head = RiskHead(latent_dim, max(8, hidden_dim // 2))
        self.stage_head = StageHead(latent_dim, max(8, hidden_dim // 2), num_stages)

    def forward(self, x_history: torch.Tensor):
        """
        x_history: (B, W, D)
        Returns pred_states (B, K, D), risk_logits (B, K), stage_logits (B, K, S)
        """
        z_curr = self.encoder(x_history)

        states, risks, stages = [], [], []
        for _ in range(self.forecast_steps):
            z_next = self.transition(z_curr)
            states.append(self.decoder(z_next))
            risks.append(self.risk_head(z_next).squeeze(-1))
            stages.append(self.stage_head(z_next))
            z_curr = z_next  # recursive latent feedback, no ground truth injected

        return (torch.stack(states, dim=1),
                torch.stack(risks, dim=1),
                torch.stack(stages, dim=1))

    @torch.no_grad()
    def predict(self, x_history: torch.Tensor):
        """Inference helper: returns (states, risk_probs (B,K), stage_probs (B,K,S))."""
        self.eval()
        states, risk_logits, stage_logits = self.forward(x_history)
        return (states,
                torch.sigmoid(risk_logits),
                torch.softmax(stage_logits, dim=-1))

    def rollout_latents(self, x_history: torch.Tensor) -> torch.Tensor:
        """Exposes the latent trajectory z_{t+1..t+K} for trajectory analysis."""
        z = self.encoder(x_history)
        zs = []
        for _ in range(self.forecast_steps):
            z = self.transition(z)
            zs.append(z)
        return torch.stack(zs, dim=1)


class VanillaLSTMClassifier(nn.Module):
    """
    Baseline 2: sequence classifier S_{t-W:t} -> P(Y_{t+1}). No latent transition,
    no multi-step rollout. Returns a LOGIT.
    """

    def __init__(self, input_dim: int = 42, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=0.1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (h_n, c_n) = self.lstm(x)
        return self.classifier(h_n[-1]).squeeze(-1)
