import torch.nn as nn
import torch


def get_latent(gen_model, x, y):
    mu, logvar = gen_model.encode(x, y)
    return mu

def gen_man_adv_samples(
    model,
    gen_model,
    x_natural,
    y,
    step_size=0.003,
    epsilon=0.031,
    perturb_steps=10,
    stats_dict=None,
):
    criterion_ce = nn.CrossEntropyLoss(reduction="mean")

    model.eval()
    gen_model.eval()

    z_natural = get_latent(gen_model, x_natural, y).detach()

    z_std = torch.as_tensor(stats_dict["std"], device=x_natural.device, dtype=x_natural.dtype)
    z_min = torch.as_tensor(stats_dict["min"], device=x_natural.device, dtype=x_natural.dtype)
    z_max = torch.as_tensor(stats_dict["max"], device=x_natural.device, dtype=x_natural.dtype)

    # feature/channel dimension is always dim=1
    num_features = z_natural.shape[1]
    # reshape stats for broadcasting
    if z_natural.ndim == 2:
        # z_natural: [B, F]
        view_shape = (1, num_features)
    else:
        # z_natural: [B, F, T]
        view_shape = (1, num_features, 1)

    z_std = torch.clamp(z_std, min=1e-3)
    z_std = z_std.view(*view_shape)
    z_min = z_min.view(*view_shape)
    z_max = z_max.view(*view_shape)

    epsilon_abs = epsilon * z_std

    if y.dim() > 1:
        y_ce = y.argmax(dim=1)
    else:
        y_ce = y.long()


    # random start inside latent epsilon-ball around z_natural
    z_adv = z_natural + (2.0 * torch.rand_like(z_natural) - 1.0) * epsilon_abs
    z_adv = torch.clamp(z_adv, z_min, z_max)

    for _ in range(perturb_steps):
        z_adv.requires_grad_()

        with torch.enable_grad():
            x_adv = gen_model.decode(z_adv, y)
            logits = model(x_adv)
            loss = criterion_ce(logits, y_ce)
            grad = torch.autograd.grad(loss, [z_adv])[0]

        z_adv = z_adv.detach() + step_size * torch.sign(grad.detach())
        # project back into per-feature epsilon box around natural latent
        z_adv = torch.min(torch.max(z_adv, z_natural - epsilon_abs), z_natural + epsilon_abs)
        z_adv = torch.clamp(z_adv, z_min, z_max)

    x_adv = gen_model.decode(z_adv, y)
    return x_adv.detach()

