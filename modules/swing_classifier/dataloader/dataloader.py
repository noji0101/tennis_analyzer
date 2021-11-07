"""DataLoader class"""

import torch.utils.data as data
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader as TorchDataLoader
import numpy as np

from modules.swing_classifier.dataloader.preprocess import preprocess_points, preprocess_anno


class DataLoader:
    """Data Loader class"""

    @staticmethod
    def load_data(points_path, anno_path, delete_weight, batch_size, mode, train_val_ratio=[8, 2], anno_delimiter=','):
        points = np.loadtxt(points_path, delimiter=',')
        try:
            targets = np.loadtxt(anno_path, delimiter=anno_delimiter)
        except ValueError:
            try:
                targets = np.loadtxt(anno_path, delimiter=anno_delimiter, skiprows=1)
            except ValueError:
                targets = np.loadtxt(anno_path)
                
        points = preprocess_points(points, delete_weight=delete_weight)
        targets = preprocess_anno(targets, points)

        if mode == 'train':
            total_frame = len(points)
            n_train = round(total_frame * 0.1 * train_val_ratio[0])

            trainset = TensorDataset(points[:n_train], targets[:n_train])
            valset = TensorDataset(points[n_train+1:], targets[n_train+1:])

            train_loader = TorchDataLoader(trainset, batch_size=batch_size, shuffle=False)
            val_loader = TorchDataLoader(valset, batch_size=batch_size, shuffle=False)

            return train_loader, val_loader

        elif mode == 'eval':
            testset = TensorDataset(points, targets)
            test_loader = TorchDataLoader(testset, batch_size=batch_size, shuffle=False)

            return test_loader

        else:
            raise ValueError('the mode should be train or eval. this mode is not supported')



