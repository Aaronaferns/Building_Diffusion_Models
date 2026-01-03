import copy
import torch

class EMA:
    """
    Maintains an exponential moving average of model parameters.
    Use ema.apply_to(model) to copy EMA weights into model for eval,
    and ema.restore(model) to restore training weights.
    """
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # initialize shadow weights
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        d = self.decay
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            assert name in self.shadow
            self.shadow[name].mul_(d).add_(p.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module):
        """Swap model params to EMA params (store current in backup)."""
        self.backup = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.backup[name] = p.detach().clone()
            p.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: torch.nn.Module):
        """Restore params that were active before apply_to()."""
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            p.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state):
        self.decay = state["decay"]
        self.shadow = state["shadow"]
