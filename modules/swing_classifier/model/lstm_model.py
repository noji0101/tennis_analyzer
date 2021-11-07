"""LSTM model"""

from modules.swing_classifier.sc_utils.paths import Paths
from modules.swing_classifier.sc_utils.logger import setup_logger, get_logger
from modules.swing_classifier.dataloader.dataloader import DataLoader
from modules.swing_classifier.executor.trainer import Trainer
from modules.swing_classifier.model.lstm.lstm import LSTM
from modules.swing_classifier.model.common.device import setup_device
from modules.swing_classifier.model.common.make_criterion import make_criterion
from modules.swing_classifier.model.common.make_optimizer import make_optimizer
from modules.swing_classifier.model.common.ckpt import load_ckpt

LOG = get_logger(__name__)

class LSTMModel():
    """LSTM Model Class"""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.model_name = self.config['model']['name']
        self.n_classes = self.config['model']['n_classes']
        self.classes = self.config['data']['classes']
        self.batch_size = self.config['train']['batch_size']
        self.resume = self.config['model']['resume']
        self.weights_path = self.config['model']['weights_path']
        self.n_gpus = self.config['train']['n_gpus']
        self.delete_weight = config['data']['delete_weight']
        self.train_val_ratio = config['data']['train_val_ratio']
        self.points_path = config['data']['points_path']
        self.anno_path = config['data']['anno_path']
        self.ckpt_dir = config['data']['ckpt_dir']
        
        
        # dataloader
        self.trainloader = None
        self.valloader = None
        self.testloader = None

        self.paths = None

    def load_data(self, is_eval):
        """Loads and Preprocess data"""
        self._set_logging()

        LOG.info(f"\nLoading {self.config['data']['points_path']} dataset...")
        # train
        if not is_eval:
            # train data
            LOG.info(f' Train data...')
            mode = 'train'
            self.trainloader, self.valloader = DataLoader().load_data(self.points_path, self.anno_path, self.delete_weight, self.batch_size, mode, self.train_val_ratio)

        # evaluation
        if is_eval:
            LOG.info(f' Test data...')
            mode = 'eval'
            self.testloader = DataLoader().load_data(self.points_path, self.anno_path, self.delete_weight, self.batch_size, mode)

    def _set_logging(self):
        """Set logging"""
        self.paths = Paths.make_dirs(self.config['util']['logdir'])
        setup_logger(str(self.paths.logdir / 'info.log'))

    def build(self, is_eval):
        """ Builds model """
        LOG.info(f'\n Building {self.model_name.upper()}...')

        if is_eval:
            phase = 'inference'
        else:
            phase = 'train'

        if self.model_name == 'SwingClassifier':
            self.model = LSTM(self.config['model'])
            if phase == 'inference':
                self.model.load_state_dict(load_ckpt(self.weights_path))
        else:
            raise ValueError('This model name is not supported.')

        # Load checkpoint
        if self.resume:
            ckpt = load_ckpt(self.resume)
            self.model.load_state_dict(ckpt['model_state_dict'], strict=False)

        LOG.info(' Model was successfully build.')

    def _set_training_parameters(self):
        """Sets training parameters"""
        self.epochs = self.config['train']['epochs']

        # CPU or GPU(single, multi)
        self.device = setup_device(self.n_gpus)
        self.model = self.model.to(self.device)

        # optimizer and criterion
        self.optimizer = make_optimizer(self.model, self.config['train']['optimizer'])
        self.criterion = make_criterion(self.config['train']['criterion'])

        # metric
        # self.metric = Metric(self.n_classes, self.classes, self.paths.metric_dir)

    def train(self):
        """Compiles and trains the model"""
        LOG.info('\n Training started.')
        self._set_training_parameters()
        
        train_parameters = {
            'device': self.device,
            'model': self.model,
            'dataloaders': (self.trainloader, self.valloader),
            'epochs': self.epochs,
            'optimizer': self.optimizer,
            'criterion': self.criterion,
            'ckpt_dir': self.ckpt_dir,
        }

        trainer = Trainer(**train_parameters)
        trainer.train()

    def evaluate(self):
        """Predicts resuts for the test dataset"""
        LOG.info('\n Prediction started...')
        self._set_training_parameters()

        eval_parameters = {
            'device': self.device,
            'model': self.model,
            'dataloaders': (self.trainloader, self.testloader),
            'epochs': None,
            'optimizer': self.optimizer,
            'criterion': self.criterion,
            # 'metrics': self.metric,
            'ckpt_dir': self.ckpt_dir,
        }

        trainer = Trainer(**eval_parameters)
        trainer.test()