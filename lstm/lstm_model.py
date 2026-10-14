from torch import nn
from torch.nn.modules.dropout import Dropout


class LSTM(nn.Module):
    """Unbounded sequence regressor for legacy price targets."""
    def __init__(self, input_size, hidden_size, num_layers, output_dim, drop=0.1, batch_first=True):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=batch_first)
        self.drop = Dropout(drop)
        self.seq = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_dim))

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.seq(self.drop(output[:, -1, :]))
