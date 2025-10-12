# Copyright 2025-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module contains the corrected implementation of the LoRA-FA optimizer
with proper device handling to prevent device mismatch errors during checkpoint loading.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Callable

import torch
import torch.nn as nn
from accelerate.utils.imports import is_bf16_available
from torch import autocast
from torch.optim import Optimizer

from ..peft_model import PeftModel
from ..utils.other import infer_device


class LoraFAOptimizer(Optimizer):
    """
    Device-safe implementation of the LoRA-FA optimizer.
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            scaling_factor = group["scaling_factor"]
            param_list = []
            name_list = []

            for p, n in zip(group["params"], group["names"]):

                # Skip no-grad parameters if not LoRA
                if "lora" not in n and p.grad is None:
                    continue

                grad = p.grad

                device = p.device

                if "lora" in n:
                    param_list.append(p)
                    name_list.append(n)
                    if len(param_list) == 2:
                        name = n[: n.find("lora")] + "lora"
                    elif len(param_list) == 1:
                        continue
                else:
                    name = n

                state = self.state[name]

                # Proper device-aware state initialization
                if len(state) == 0:
                    if len(param_list) == 2:
                        # LoRA pair
                        state["step"] = 0
                        state["exp_avg_B"] = torch.zeros_like(param_list[1], device=device)
                        state["exp_avg_sq_B"] = torch.zeros_like(param_list[1], device=device)
                    else:
                        # Normal parameter
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p, device=device)
                        state["exp_avg_sq"] = torch.zeros_like(p, device=device)

                # --- LoRA-FA logic ---
                if len(param_list) == 2:
                    A = param_list[0]
                    B = param_list[1]
                    grad_B_orin = B.grad

                    delta = 1e-8
                    AA_T = A @ A.T
                    AA_T_inv = torch.linalg.pinv(
                        AA_T + delta * torch.eye(A.shape[0]).to(A.device)
                    )

                    device_type = infer_device()

                    if is_bf16_available():
                        with autocast(device_type=device_type, dtype=torch.bfloat16):
                            grad_B = (1 / scaling_factor**2) * (grad_B_orin @ AA_T_inv)
                    else:
                        grad_B = (1 / scaling_factor**2) * (grad_B_orin @ AA_T_inv)

                    if grad_B.dtype != B.grad.dtype:
                        grad_B = grad_B.to(B.grad.dtype)

                    exp_avg_B = state["exp_avg_B"]
                    exp_avg_sq_B = state["exp_avg_sq_B"]
                    beta1, beta2 = group["betas"]

                    state["step"] += 1
                    exp_avg_B.mul_(beta1).add_(grad_B, alpha=(1.0 - beta1))
                    exp_avg_sq_B.mul_(beta2).addcmul_(grad_B, grad_B, value=1.0 - beta2)

                    denom_B = exp_avg_sq_B.sqrt().add_(group["eps"])
                    step_size = group["lr"]
                    if group["correct_bias"]:
                        bias_correction1 = 1.0 - beta1 ** state["step"]
                        bias_correction2 = 1.0 - beta2 ** state["step"]
                        step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                    B.addcdiv_(exp_avg_B, denom_B, value=-step_size)

                    if group["weight_decay"] > 0.0:
                        B.add_(B, alpha=(-group["lr"] * group["weight_decay"]))

                    param_list = []
                    name_list = []

                # --- Standard AdamW path ---
                else:
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    beta1, beta2 = group["betas"]

                    state["step"] += 1

                    exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                    step_size = group["lr"]
                    if group["correct_bias"]:
                        bias_correction1 = 1.0 - beta1 ** state["step"]
                        bias_correction2 = 1.0 - beta2 ** state["step"]
                        step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                    p.addcdiv_(exp_avg, denom, value=-step_size)

                    if group["weight_decay"] > 0.0:
                        p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        return loss


def create_lorafa_optimizer(
    model: PeftModel, r: int, lora_alpha: int, lr: float, weight_decay: float = 0.0, use_rslora: bool = False
) -> Optimizer:
    """
    Creates a LoraFAOptimizer instance configured for the given PEFT model,
    with proper device-safe initialization.
    """
    # Freeze LoRA A matrices
    for name, param in model.named_parameters():
        if "lora_A" in name:
            param.requires_grad_(False)

    lora_scaling = lora_alpha / math.sqrt(r) if use_rslora else lora_alpha / r

    named_params = list(model.named_parameters())

    param_groups = [
        {
            "params": [p for _, p in named_params],
            "names": [name for name, _ in named_params],
            "lr": lr,
            "scaling_factor": lora_scaling,
            "betas": (0.9, 0.999),
            "weight_decay": weight_decay,
        }
    ]

    optimizer = LoraFAOptimizer(param_groups)

    # Ensure all optimizer states are placed on the same device as parameters (for checkpoint safety)
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p is not None and p.device:
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            optimizer.state[name][k] = v.to(p.device)

    return optimizer
