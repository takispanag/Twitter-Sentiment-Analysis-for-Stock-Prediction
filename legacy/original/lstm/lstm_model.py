from torch import nn
from torch.nn.modules.dropout import Dropout


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_dim, drop=0.1, batch_first=True):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first)
        self.drop = Dropout(drop)
        self.seq = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.drop(output)
        output = self.seq(output)
        return output
