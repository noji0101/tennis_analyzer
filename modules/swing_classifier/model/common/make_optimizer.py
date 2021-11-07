"""Make Optimizer"""
import torch.optim as optim

from utils.logger import get_logger

LOG = get_logger(__name__)

def make_optimizer(model: object, optimizer_cfg: object) -> object:
    if optimizer_cfg['type'] == 'sgd':
        LOG.info('\n Optimizer: SGD')
        return optim.SGD(model.parameters(), 
                         lr=optimizer_cfg['lr'],
                         momentum=optimizer_cfg['momentum'],
                         weight_decay=optimizer_cfg['decay'])
    elif optimizer_cfg['type'] == 'adam':
        LOG.info('\n Optimizer: Adam')
        return optim.Adam(model.parameters(),
                          lr=optimizer_cfg['lr'],
                          betas=(optimizer_cfg['beta1'], optimizer_cfg['beta2']),
                          eps=optimizer_cfg['eps'])
    elif optimizer_cfg['type'] == 'amsgrad':
        LOG.info('\n Optimizer: AMSGrad')
        return optim.Adam(model.parameters(),
                          lr=optimizer_cfg['lr'], 
                          betas=(optimizer_cfg['beta1'], optimizer_cfg['beta2']), 
                          eps=optimizer_cfg['eps'], 
                          amsgrad=True)
    elif optimizer_cfg['type'] == 'rmsprop':
        LOG.info('\n Optimizer: RMSProp')
        return optim.RMSprop(model.parameters(), 
                             lr=optimizer_cfg['lr'], 
                             alpha=optimizer_cfg['alpha'], 
                             eps=optimizer_cfg['eps'])
    elif optimizer_cfg['type'] == "adadelta":
        LOG.info('\n Optimizer: Adadelta')
        return optim.Adadelta(model.parameters(),
                              lr=optimizer_cfg['lr'],
                              rho=optimizer_cfg['rho'],
                              eps=optimizer_cfg['eps'],
                              weight_decay=optimizer_cfg['decay'])
    elif optimizer_cfg['type'] == "adagrad":
        LOG.info('\n Optimizer: Adagrad')
        return optim.Adagrad(model.parameters(),
                             lr=optimizer_cfg['lr'],
                             lr_decay=optimizer_cfg['lr_decay'],
                             weight_decay=optimizer_cfg['decay'],
                             initial_accumulator_value=optimizer_cfg['initial_accumulator_value'])
    else:
        raise NotImplementedError('This optimizer is not supported.')