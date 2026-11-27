"""Modify standard PyTorch distributions so they to make compatible with this codebase."""
import torch
import torch.nn as nn
from harl.utils.models_tools import init, get_init_method


class FixedCategorical(torch.distributions.Categorical):
    """Modify standard PyTorch Categorical."""

    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class Categorical(nn.Module):
    """A linear layer followed by a Categorical distribution."""

    def __init__(
        self, num_inputs, num_outputs, initialization_method="orthogonal_", gain=0.01
    ):
        super(Categorical, self).__init__()
        init_method = get_init_method(initialization_method)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None):
        x = self.linear(x)
        if available_actions is not None:
            x[available_actions == 0] = -1e10
        return FixedCategorical(logits=x)


class OrderedGaussian(nn.Module):
    """Categorical probabilities from a learned center and width in bin units."""

    def __init__(self, num_inputs, num_outputs, initialization_method, gain):
        super().__init__()
        if num_outputs < 2:
            raise ValueError("Ordered discrete control requires at least two bins")
        self.last_bin = num_outputs - 1
        self.parameters_head = init(nn.Linear(num_inputs, 2), get_init_method(initialization_method),
                                    lambda bias: nn.init.constant_(bias, 0), gain)
        self.register_buffer("bins", torch.arange(num_outputs, dtype=torch.float32), persistent=False)

    def forward(self, x, available_actions=None):
        center_fraction, width_fraction = self.parameters_head(x).sigmoid().split(1, dim=-1)
        mu = self.last_bin * center_fraction
        sigma = 0.5 + 0.5 * self.last_bin * width_fraction
        logits = -0.5 * ((self.bins - mu) / sigma).square()
        if available_actions is not None:
            if not (available_actions != 0).any(dim=-1).all():
                raise ValueError("Each observation must have an available action")
            logits = logits.masked_fill(available_actions == 0, -torch.inf)
        return FixedCategorical(logits=logits)


ACTION_HEADS = {"categorical": Categorical, "ordered_gaussian": OrderedGaussian}
