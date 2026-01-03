import torch
from tqdm import tqdm

NUM_TIMESTEPS = 1000

BETA__T = torch.linspace(1e-4, 0.02, NUM_TIMESTEPS)
ALPHA__T = 1.0 - BETA__T
ALPHA_BAR__T = torch.cumprod(ALPHA__T, dim=0)

def forwardDiffusion(X0__BCHW):
    device = X0__BCHW.device
    beta_T = BETA__T.to(device)
    alpha_T = ALPHA__T.to(device)
    alpha_bar_T = ALPHA_BAR__T.to(device)

    B = X0__BCHW.shape[0]
    eps__BCHW = torch.randn_like(X0__BCHW)
    t__B = torch.randint(low=1, high=NUM_TIMESTEPS+1, size=(B,), device=device, dtype=torch.long) # t in [1,1000]

    ab = alpha_bar_T[t__B - 1][:, None, None, None] #because of 0 indicing
    Xt_BCHW = torch.sqrt(ab) * X0__BCHW + torch.sqrt(1.0 - ab) * eps__BCHW
    return eps__BCHW, t__B, Xt_BCHW

def diffusionTrainStep(model, optimizer, loss_fn, X0, device):
    X0__BCHW = X0.to(device)
    eps__BCHW, t__B, Xt_BCHW = forwardDiffusion(X0__BCHW)
    eps_pred__BCHW = model(Xt_BCHW, t__B)
    loss = loss_fn(eps_pred__BCHW, eps__BCHW)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def sample(model, B, C, H, W, device=None):
    model.eval()
    device = next(model.parameters()).device if device is None else device

    beta_T = BETA__T.to(device)
    alpha_T = ALPHA__T.to(device)
    alpha_bar_T = ALPHA_BAR__T.to(device)

    x = torch.randn((B, C, H, W), device=device)

    for t in range(NUM_TIMESTEPS, 0, -1):
        t__B = torch.full((B,), t, device=device, dtype=torch.long)
        eps_pred = model(x, t__B)

        alpha_t = alpha_T[t__B - 1][:, None, None, None]
        beta_t  = beta_T[t__B - 1][:, None, None, None]
        ab_t    = alpha_bar_T[t__B - 1][:, None, None, None]

        mean = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1.0 - ab_t)) * eps_pred)

        if t > 1:
            ab_prev = alpha_bar_T[t__B - 2][:, None, None, None]
            beta_tilde = ((1.0 - ab_prev) / (1.0 - ab_t)) * beta_t
            z = torch.randn_like(x)
            x = mean + torch.sqrt(beta_tilde) * z
        else:
            x = mean

    return x.detach().cpu()