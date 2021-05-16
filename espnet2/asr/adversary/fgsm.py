# Copyright 2020 Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


"""FastGradientSignMethod attack definition."""


class FastGradientSignMethod(object):
    """One step FastGradientSignMethod (Goodfellow et al, 2014)

    Paper: https://arxiv.org/abs/1412.6572
    """

    def __init__(
        self,
        adv_begin_epoch=0,
        adv_prob=1.0,
        adv_lr=0.1,
        adv_weight=0.3,
    ):
        """Construct a FastGradientSignMethod object.

        Args:
            adv_begin_epoch (int): Begining epoch to conducti adversarial training. Defaults to 0.
            adv_prob (float): Probability to conducti adversarial training. Defaults to 1.0.
            adv_lr (float): Learning rate of adversarial perturbation. Defaults to 0.1.
            adv_weight (float): Weight of adversarial loss. Defaults to 0.3.
            no_sync_dp (bool): [description]. Defaults to False. Defaults to False.
        """

        super().__init__()
        self.name = "fgsm"
        self.adv_begin_epoch = adv_begin_epoch
        self.adv_prob = adv_prob
        self.adv_lr = adv_lr
        self.adv_weight = adv_weight

    def __call__(self, model):
        return model
