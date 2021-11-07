import os
from tqdm  import tqdm

import numpy as np
import torch
from torch import nn


class Trainer():

    def __init__(self, **kwargs):
        self.device = kwargs['device']
        self.model = kwargs['model']
        self.trainloader, self.testloader = kwargs['dataloaders']
        self.dataloaders_dict = {"train": self.trainloader, "val": self.testloader}
        self.epochs = kwargs['epochs']
        self.optimizer = kwargs['optimizer']
        self.criterion = kwargs['criterion']
        self.ckpt_dir = kwargs['ckpt_dir']

    def train(self):

        train_accuracies = []
        train_losses = []
        val_accuracies = []
        val_losses = []

        best_loss = np.inf

        for epoch in range(self.epochs + 1):
            print(f'\n==================== Epoch: {epoch} ====================')
            print('\n Train:')

            epoch_train_loss = 0.0
            n_total = 0
            n_correct = 0
            
            self.model.train()
            with tqdm(self.trainloader, ncols=100) as pbar:
                for i, (points_batch, targets_batch) in enumerate(pbar):
                    points_batch = points_batch.to(self.device)
                    targets_batch = targets_batch.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(points_batch)
                    loss = self.criterion(outputs, targets_batch)
                    loss.backward()
                    self.optimizer.step()

                    epoch_train_loss += loss.item()

                    preds = outputs.argmax(axis=1)
                    n_total += targets_batch.size(0)
                    n_correct += (preds == targets_batch).sum().item()

            epoch_train_accuracy = 100.0 * n_correct / n_total
            epoch_train_loss = epoch_train_loss / i
            epoch_val_loss, epoch_val_accuracy = self.test()

            train_accuracies.append(epoch_train_accuracy)
            train_losses.append(epoch_train_loss)
            val_accuracies.append(epoch_val_accuracy)
            val_losses.append(epoch_val_loss)

            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                print(' Saving Best Checkpoint...')
                self._save_ckpt(epoch)

    def test(self):
        test_loss = 0.0
        n_total = 0
        n_correct = 0
        
        self.model.eval()
        with torch.no_grad():
            with tqdm(self.testloader, ncols=100) as pbar:
                for i, (points_batch, targets_batch) in enumerate(pbar):
                    points_batch = points_batch.to(self.device)
                    targets_batch = targets_batch.to(self.device)

                    outputs = self.model(points_batch)
                    loss = self.criterion(outputs, targets_batch)

                    test_loss += loss.item()

                    preds = outputs.argmax(axis=1)
                    n_total += targets_batch.size(0)
                    n_correct += (preds == targets_batch).sum().item()

        test_accuracy = 100.0 * n_correct / n_total
        test_loss = test_loss / i
        print(f'test_loss:{test_loss}\ntest_accuracy:{test_accuracy}')

        return test_loss, test_accuracy
    
    def _save_ckpt(self, epoch):
        if isinstance(self.model, nn.DataParallel):
            model = self.model.module
        else:
            model = self.model
        
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
    
        ckpt_path = self.ckpt_dir + f'/best_acc_ckpt_{epoch}.pth'


        torch.save({
            'epoch': epoch,
            'model': model,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, ckpt_path)