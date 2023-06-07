from __future__ import division
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from normal_ranking_loss import EdgeguidedNormalRankingLoss


device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")



normal_ranking_loss = EdgeguidedNormalRankingLoss().to(device)