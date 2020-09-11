import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask


class FreeLB(object):
    """Free large batch adversarial training (Zhu et al, 2020)
    Paper: https://arxiv.org/pdf/1909.11764.pdf
    """

    def __init__(
        self,
        model,
        adv_prob=1.0,
        adv_weight=0.0,
        adv_alpha=0.025,
        adv_step=3,
        adv_init_bound=0.4,
        adv_begin_epoch=-1,
        adv_max_norm=0.3,
        no_sync_dp=False,
        norm_method="linf",
    ):
        """Create an instance of the FreeLB attack."""
        self.model = model
        self.adv_prob = adv_prob
        self.adv_weight = adv_weight
        self.adv_alpha = adv_alpha
        self.adv_step = adv_step
        self.adv_init_bound = adv_init_bound
        self.adv_begin_epoch = adv_begin_epoch
        self.adv_max_norm = adv_max_norm
        self.no_sync_dp = no_sync_dp
        self.norm_method = norm_method

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
        if self.adv_init_bound > 0.0 and epoch >= self.adv_begin_epoch:
            # init adversarial noise under L2 normalization
            x_mask = make_non_pad_mask(ilens.tolist()).to(x.device).float()
            if self.norm_method == "l2":
                delta = torch.zeros_like(x).uniform_(-1, 1) * x_mask.unsqueeze(2)
                dims = ilens.to(delta) * x.size(-1)
                # ilens >> NLP's ilens, dims will be larger and mag will be smaller?
                mag = self.adv_init_bound / torch.sqrt(dims)
                delta = (delta * mag.view(-1, 1, 1)).detach()
            elif self.norm_method == "linf":
                delta = torch.zeros_like(x).uniform_(
                    -self.adv_init_bound, self.adv_init_bound
                ) * x_mask.unsqueeze(2)
            else:
                raise ValueError("Not implemented.")
            delta.requires_grad_()

            loss_adv = self.model(x + delta, ilens, y, init_dp=True) / accum_grad
        else:
            delta = 0
            loss_adv = self.model(x, ilens, y, init_dp=True) / accum_grad

        if epoch >= self.adv_begin_epoch:
            loss_adv = loss_adv / (1 + self.adv_step)

        loss_adv.backward()

        if self.adv_init_bound > 0.0 and epoch >= self.adv_begin_epoch:
            delta_grad = delta.grad.clone().detach()
        else:
            delta_grad = x.grad.clone().detach()

        # update adversarial noise
        for step in range(self.adv_step):
            if self.norm_method == "l2":
                denorm = torch.clamp(
                    torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(
                        -1, 1, 1
                    ),
                    min=1e-10,
                )
                delta = (delta + self.adv_alpha * delta_grad / denorm).detach()
                if self.adv_max_norm > 0.0:
                    delta_norm = (
                        torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1)
                        .to(x)
                        .detach()
                    )
                    exceed_mask = (delta_norm > self.adv_max_norm).to(x)
                    delta = (
                        delta
                        * (
                            self.adv_max_norm / delta_norm * exceed_mask
                            + (1 - exceed_mask)
                        )
                        .view(-1, 1, 1)
                        .detach()
                    )
            elif self.norm_method == "linf":
                denorm = torch.norm(
                    delta_grad.view(delta.grad.size(0), -1), dim=1, p=float("inf")
                ).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + self.adv_alpha * delta_grad / denorm).detach()
                if self.adv_max_norm > 0:
                    delta = torch.clamp(
                        delta, -self.adv_max_norm, self.adv_max_norm
                    ).detach()
            else:
                raise ValueError("Not implemented.")

            delta.requires_grad_()

            loss_adv = (
                self.model(x + delta, ilens, y, init_dp=self.no_sync_dp, record=False)
                / accum_grad
            )

            if step == self.adv_step - 1 and self.adv_weight > 0.0:
                logits_adv = self.model.pred_pad
                _ = self.model(x, ilens, y, init_dp=False, record=False) / accum_grad
                logits = self.model.pred_pad
                loss_reg = self.sym_kld(logits, logits_adv) / accum_grad
                loss_adv = loss_adv + self.adv_weight * loss_reg

            loss_adv = loss_adv / (1 + self.adv_step)
            loss_adv.backward()
            delta_grad = delta.grad.clone().detach()
