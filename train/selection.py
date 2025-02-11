import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

def selection(models:list[nn.Module], params:dict, train:torch.tensor, val:torch.tensor):
    return NotImplementedError