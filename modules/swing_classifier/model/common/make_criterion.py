"""Make Criterion"""
import torch.nn as nn

from utils.logger import get_logger
from modules.swing_classifier.model.criterion import CustomLoss

LOG = get_logger(__name__)

def make_criterion(criterion_cfg):
    if criterion_cfg['type'] == 'CustomLoss':
        LOG.info('\n Criterion: CustomLoss')
        return CustomLoss(swing_ratio=criterion_cfg['swing_ratio'])
    else:
        raise NotImplementedError('This loss function is not supported.')