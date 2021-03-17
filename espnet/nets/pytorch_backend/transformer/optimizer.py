#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Optimizer module."""

import torch


class NoamOpt(object):
    """Optim wrapper that implements rate."""

    def __init__(self, model_size, factor, warmup, optimizer):
        """Construct an NoamOpt object."""
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        """Update parameters and rate."""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """Implement `lrate` above."""
        if step is None:
            step = self._step
        return (
            self.factor
            * self.model_size ** (-0.5)
            * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )

    def zero_grad(self):
        """Reset gradient."""
        self.optimizer.zero_grad()

    def state_dict(self):
        """Return state_dict."""
        return {
            "_step": self._step,
            "warmup": self.warmup,
            "factor": self.factor,
            "model_size": self.model_size,
            "_rate": self._rate,
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """Load state_dict."""
        for key, value in state_dict.items():
            if key == "optimizer":
                self.optimizer.load_state_dict(state_dict["optimizer"])
            else:
                setattr(self, key, value)


def get_std_opt(model_params, d_model, warmup, factor):
    """Get standard NoamOpt."""
    base = torch.optim.Adam(model_params, lr=0, betas=(0.9, 0.98), eps=1e-9)
    return NoamOpt(d_model, factor, warmup, base)


class OneCycleLR(object):
    """Optim wrapper that implements OneCycleLR scheduler."""

    def __init__(self, optimizer, scheduler):
        """Construct an OneCycleLR scheduler wrapper."""
        self.optimizer = optimizer
        self.scheduler = scheduler

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        """Update paramters and learning rates."""
        self.optimizer.step()
        self.scheduler.step()

    def zero_grad(self):
        """Reset gradients."""
        self.optimizer.zero_grad()

    def state_dict(self):
        """Return state_dict."""
        return {
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """Load state_dict."""
        for key, value in state_dict.items():
            if key == "optimizer":
                self.optimizer.load_state_dict(state_dict["optimizer"])
            else:
                setattr(self, key, value)


def get_onecycle_opt(
    model_params,
    max_lr,
    steps_per_epoch,
    epochs,
    pct_start=0.3,
    final_div_factor=1e4,
    last_epoch=-1,
):
    """Get stadard OneCycleLR scheduler.

    Args:
        model_params ([type]): [description]
        max_lr ([type]): [description]
        steps_per_epoch ([type]): [description]
        epochs (int):
        pct_start (float, optional): [description]. Defaults to 0.3.
        final_div_factor ([type], optional): [description]. Defaults to 1e4.
        last_epoch (int, optional): [description]. Defaults to -1.
    """
    # optimizer = torch.optim.SGD(model_params, lr=0.1, momentum=0.9)
    optimizer = torch.optim.Adam(model_params, lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=pct_start,
        final_div_factor=final_div_factor,
        last_epoch=last_epoch,
    )

    return OneCycleLR(optimizer, scheduler)
