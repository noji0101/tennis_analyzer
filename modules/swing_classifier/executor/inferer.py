import torch
import numpy as np

from modules.swing_classifier.model.common.device import setup_device
from modules.swing_classifier.model import get_model
from modules.swing_classifier.dataloader.preprocess import preprocess_points


class SwingInferer():

    def __init__(self, points, config):
        self.points = points
        self.delete_weight = config['data']['delete_weight']
        self.n_gpus = config['train']['n_gpus']
        self.device = setup_device(self.n_gpus)

        # self.classes = self.model.classes
        self.model = get_model(config)
        self.model.build(is_eval=True)
        self.net = self.model.model

        # 重みの読み込み
        self.weights_path = config['model']['weights_path']
        self.net.load_state_dict(torch.load(self.weights_path))

        # device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.net.to(self.device)
        self.net.eval()

    def predict(self):
        print('predicting swing...')
        points = preprocess_points(self.points, self.delete_weight)
        with torch.no_grad():
            # infer
            points = points.to(self.device)
            outputs = self.net(points)
            y_pred = np.argmax(outputs.cpu().detach().numpy(), axis=1)
        print('done')
        return y_pred