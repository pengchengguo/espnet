import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask


class FreeLB(object):
    """Free large batch adversarial training (Zhu et al, 2020)

    Paper: https://arxiv.org/pdf/1909.11764.pdf

    Args:
        model (torch.nn.Module): the model to be trained
        adv_prob (float):
        adv_lr (float):
        adv_steps (float):
        rand_init_mag (float):
        adv_begin_epoch (int):
        no_sync_dp (bool):
        max_norm (float):
        norm_method (str):
        reg_weight (float):
    """

    def __init__(
        self,
        model,
        adv_prob=1.0,
        adv_lr=0.025,
        adv_steps=3,
        rand_init_mag=0.4,
        adv_begin_epoch=-1,
        no_sync_dp=False,
        max_norm=0.3,
        norm_method="l2",
        reg_weight=0.0,
    ):
        """Create an instance of the FreeLB adversary."""
        self.model = model
        self.adv_prob = adv_prob
        self.adv_lr = adv_lr
        self.adv_steps = adv_steps
        self.rand_init_mag = rand_init_mag
        self.adv_begin_epoch = adv_begin_epoch
        self.no_sync_dp = no_sync_dp
        self.max_norm = max_norm
        self.norm_method = norm_method
        self.reg_weight = reg_weight

    def sym_kld(self, net_out_1, net_out_2):
        P = F.softmax(net_out_1, dim=-1, dtype=torch.float32)
        Q = F.softmax(net_out_2, dim=-1, dtype=torch.float32)

        logP = F.log_softmax(net_out_1, dim=-1, dtype=torch.float32)
        logQ = F.log_softmax(net_out_2, dim=-1, dtype=torch.float32)

        # taking sum directly, since the reduction method is sum
        sym_kld = 0.5 * torch.sum((P - Q) * (logP - logQ))

        return sym_kld

    def __call__(self, x, ilens, y, epoch, accum_grad):
        """Forward."""
        # import pdb

        # pdb.set_trace()
        self.model.train()

        x.requires_grad_()
        if self.rand_init_mag > 0.0 and epoch >= self.adv_begin_epoch:
            # init adversarial noise under L2 normalization
            x_mask = make_non_pad_mask(ilens.tolist()).to(x.device).float()
            if self.norm_method == "l2":
                delta = torch.zeros_like(x).uniform_(-1, 1) * x_mask.unsqueeze(2)
                dims = ilens.to(delta) * x.size(-1)
                # ilens >> NLP's ilens, dims will be larger and mag will be smaller?
                mag = self.rand_init_mag / torch.sqrt(dims)
                delta = (delta * mag.view(-1, 1, 1)).detach()
            elif self.norm_method == "linf":
                delta = torch.zeros_like(x).uniform_(
                    -self.rand_init_mag, self.rand_init_mag
                ) * x_mask.unsqueeze(2)
            else:
                raise ValueError("Not implemented.")
            delta.requires_grad_()

            loss_adv = self.model(x + delta, ilens, y, init_dp=True) / accum_grad
        else:
            delta = 0
            loss_adv = self.model(x, ilens, y, init_dp=True) / accum_grad

        if epoch >= self.adv_begin_epoch:
            loss_adv = loss_adv / (1 + self.adv_steps)

        loss_adv.backward()

        if self.rand_init_mag > 0.0 and epoch >= self.adv_begin_epoch:
            delta_grad = delta.grad.clone().detach()
        else:
            delta_grad = x.grad.clone().detach()

        # update adversarial noise
        for step in range(self.adv_steps):
            if self.norm_method == "l2":
                denorm = torch.clamp(
                    torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(
                        -1, 1, 1
                    ),
                    min=1e-10,
                )
                delta = (delta + self.adv_lr * delta_grad / denorm).detach()
                if self.max_norm > 0.0:
                    delta_norm = (
                        torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1)
                        .to(x)
                        .detach()
                    )
                    exceed_mask = (delta_norm > self.max_norm).to(x)
                    delta = (
                        delta
                        * (self.max_norm / delta_norm * exceed_mask + (1 - exceed_mask))
                        .view(-1, 1, 1)
                        .detach()
                    )
            elif self.norm_method == "linf":
                denorm = torch.norm(
                    delta_grad.view(delta.grad.size(0), -1), dim=1, p=float("inf")
                ).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + self.adv_lr * delta_grad / denorm).detach()
                if self.max_norm > 0:
                    delta = torch.clamp(delta, -self.max_norm, self.max_norm).detach()
            else:
                raise ValueError("Not implemented.")

            delta.requires_grad_()

            loss_adv = (
                self.model(x + delta, ilens, y, init_dp=self.no_sync_dp, record=False)
                / accum_grad
            )

            if step == self.adv_steps - 1 and self.reg_weight > 0.0:
                logits_adv = self.model.pred_pad
                _ = (
                    self.model(x, ilens, y, init_dp=self.no_sync_dp, record=False)
                    / accum_grad
                )
                logits = self.model.pred_pad
                loss_reg = self.sym_kld(logits, logits_adv) / accum_grad
                loss_adv = (1 - self.reg_weight) * loss_adv + self.reg_weight * loss_reg

            loss_adv = loss_adv / (1 + self.adv_steps)
            loss_adv.backward()
            delta_grad = delta.grad.clone().detach()
