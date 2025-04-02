from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from recovlm.training.parallel import get_sequence_parallel_world_size, get_sequence_parallel_group



class PairWiseLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    """

    def forward(
        self, 
        chosen_reward: torch.Tensor, 
        reject_reward: torch.Tensor, 
        margin: torch.Tensor = None
    ) -> torch.Tensor:
        if margin is not None:
            loss = -F.logsigmoid(chosen_reward - reject_reward - margin)
        else:
            loss = -F.logsigmoid(chosen_reward - reject_reward)
        return loss.mean()


class LogExpLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    Details: https://arxiv.org/abs/2204.05862
    """

    def forward(
        self, 
        chosen_reward: torch.Tensor, 
        reject_reward: torch.Tensor, 
        margin: torch.Tensor = None
    ) -> torch.Tensor:
        loss = torch.log(1 + torch.exp(reject_reward - chosen_reward))
        return loss.mean()
