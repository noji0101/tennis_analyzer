import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        self.n_points = config['n_points']
        self.n_hidden = config['n_hidden']
        self.n_target = config['n_target']

        self.lstm = nn.LSTM(self.n_points, self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, self.n_target)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, points):
        lstm1, _ = self.lstm(points)
        # ３次元テンソルを2次元に調整して全結合。
        x1 = self.linear(lstm1.view(-1, self.n_hidden))
        # softmaxに食わせて、確率として表現
        out = self.softmax(x1)
        return out