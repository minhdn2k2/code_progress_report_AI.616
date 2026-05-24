import torch


class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization wrapper.

    This follows the common PyTorch implementation pattern:
    1) compute gradient at w
    2) perturb w -> w + e(w)
    3) compute gradient at w + e(w)
    4) restore w and apply the base optimizer step

    The perturbation is normalized:
        e(w) = rho * grad / ||grad||_2
    """

    def __init__(self, params, base_optimizer_cls, rho=0.05, adaptive=False, **kwargs):
        if rho < 0:
            raise ValueError(f"Invalid rho: {rho}")

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norms = []

        for group in self.param_groups:
            adaptive = group["adaptive"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                if adaptive:
                    norms.append((torch.abs(p) * p.grad).norm(p=2).to(shared_device))
                else:
                    norms.append(p.grad.norm(p=2).to(shared_device))

        if not norms:
            return torch.tensor(0.0, device=shared_device)

        return torch.norm(torch.stack(norms), p=2)

    @torch.no_grad()
    def first_step(self, zero_grad=True):
        grad_norm = self._grad_norm()

        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            adaptive = group["adaptive"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if adaptive:
                    e_w = torch.pow(p, 2) * p.grad * scale
                else:
                    e_w = p.grad * scale

                p.add_(e_w)
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad=True):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                e_w = self.state[p].pop("e_w", None)
                if e_w is not None:
                    p.sub_(e_w)

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad(set_to_none=True)

    def step(self, closure=None):
        raise RuntimeError(
            "SAM requires two forward/backward passes. "
            "Use first_step() and second_step() explicitly."
        )


@torch.no_grad()
def grad_norm(parameters, adaptive=False):
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return torch.tensor(0.0)

    device = params[0].device
    norms = []
    for p in params:
        if adaptive:
            norms.append((torch.abs(p) * p.grad).norm(p=2).to(device))
        else:
            norms.append(p.grad.norm(p=2).to(device))
    return torch.norm(torch.stack(norms), p=2)


@torch.no_grad()
def perturb_weights(model, rho=0.05, adaptive=False):
    """
    Apply SAM perturbation to model parameters.

    Returns a list of (parameter, perturbation) pairs so that the caller can
    restore the original parameters after computing the second gradient.
    """
    norm = grad_norm(model.parameters(), adaptive=adaptive)
    perturbations = []

    for p in model.parameters():
        if p.grad is None:
            continue

        scale = rho / (norm + 1e-12)
        if adaptive:
            e_w = torch.pow(p, 2) * p.grad * scale
        else:
            e_w = p.grad * scale

        p.add_(e_w)
        perturbations.append((p, e_w))

    return perturbations


@torch.no_grad()
def restore_weights(perturbations):
    for p, e_w in perturbations:
        p.sub_(e_w)