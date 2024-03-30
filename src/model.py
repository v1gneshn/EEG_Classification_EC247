import torch
import torch.nn as nn


class DCNN_elu(torch.nn.Module):
    def __init__(self, n_output=4):
        super(DeepConvNet_ELU, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1,5),bias=False),
            nn.Conv2d(25, 25, kernel_size=(2,1),bias=False),
            nn.BatchNorm2d(25, eps=1e-05, momentum=0.1),
            nn.ELU(alpha=0.4),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.4),

            nn.Conv2d(25, 50, kernel_size=(1,5),bias=False),
            nn.BatchNorm2d(50, eps=1e-05, momentum=0.1),
            nn.ELU(alpha=0.4),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.4),

            nn.Conv2d(50, 100, kernel_size=(1,5),bias=False),
            nn.BatchNorm2d(100, eps=1e-05, momentum=0.1),
            nn.ELU(alpha=0.4),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.4),

            nn.Conv2d(100, 200, kernel_size=(1,5),bias=False),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.1),
            nn.ELU(alpha=0.4),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.4),

            nn.Flatten(),
            # nn.Linear(243600,n_output,bias=True)
            # nn.Linear(88200,n_output,bias=True)
            # nn.Linear(46200,n_output,bias=True)
            nn.Linear(113400,n_output,bias=True)
        )

    def forward(self, x):
        out = self.model(x)
        return out



class DCNN_relu(torch.nn.Module):
    def __init__(self, n_output=4):
        super(DeepConvNet_ReLU, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1,5),bias=False),
            nn.Conv2d(25, 25, kernel_size=(2,1),bias=False),
            nn.BatchNorm2d(25, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.47),

            nn.Conv2d(25, 50, kernel_size=(1,5),bias=False),
            nn.BatchNorm2d(50, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.47),

            nn.Conv2d(50, 100, kernel_size=(1,5),bias=False),
            nn.BatchNorm2d(100, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.47),

            nn.Conv2d(100, 200, kernel_size=(1,5),bias=False),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.47),

            nn.Flatten(),
            # nn.Linear(8600,n_output,bias=True)
            nn.Linear(113400,n_output,bias=True)
        )

    def forward(self, x):
        out = self.model(x)
        return out

class DCNN_lrelu(torch.nn.Module):
    def __init__(self, n_output=4):
        super(DeepConvNet_LeakyReLU, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1,5),bias=False),
            nn.Conv2d(25, 25, kernel_size=(2,1),bias=False),
            nn.BatchNorm2d(25, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(negative_slope=0.04),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.47),
            nn.Conv2d(25, 50, kernel_size=(1,5),bias=False),
            nn.BatchNorm2d(50, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(negative_slope=0.09),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.47),
            nn.Conv2d(50, 100, kernel_size=(1,5),bias=False),
            nn.BatchNorm2d(100, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(negative_slope=0.04),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.47),
            nn.Conv2d(100, 200, kernel_size=(1,5),bias=False),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(negative_slope=0.09),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.47),

            nn.Flatten(),
            # nn.Linear(8600,n_output,bias=True)
            nn.Linear(113400,n_output,bias=True)
        )

    def forward(self, x):
        out = self.model(x)
        return out

class CNNGRU(nn.Module):
    '''
    CNN + GRU
    '''
    def __init__(self, cnn_input_size, rnn_input_size, hidden_size, output_dim, dropout):
        super(CNNGRUnet, self).__init__()
        self.cnn_input_size = cnn_input_size
        self.rnn_input_size = rnn_input_size
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.cnn = nn.Sequential(
            nn.Conv1d(cnn_input_size, rnn_input_size, kernel_size=10, stride=2),
            nn.BatchNorm1d(rnn_input_size),
            nn.ReLU(),
        )
        self.rnn = nn.GRU(input_size=rnn_input_size, hidden_size=hidden_size, num_layers=2, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x, h=None):
        out = self.cnn(x)
        out = out.permute(2,0,1)
        if type(h) == type(None):
            out, hn = self.rnn(out)
        else:
            out, hn = self.rnn(out, h.detach())
        out = self.fc(out[-1, :, :])
        return out
class CNNLSTM(nn.Module):
    '''
    CNN + LSTM
    '''
    def __init__(self, cnn_input_size, rnn_input_size, hidden_size, output_dim, dropout):
        super(CNNLSTMnet, self).__init__()
        self.cnn_input_size = cnn_input_size
        self.rnn_input_size = rnn_input_size
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.cnn = nn.Sequential(
            nn.Conv1d(cnn_input_size, rnn_input_size, kernel_size=10, stride=2),
            nn.BatchNorm1d(rnn_input_size),
            nn.ReLU(),
        )
        self.rnn = nn.LSTM(input_size=rnn_input_size, hidden_size=hidden_size, num_layers=2, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x, h=None):
        out = self.cnn(x)
        out = out.permute(2,0,1)
        if type(h) == type(None):
            out, hn = self.rnn(out)
        else:
            out, hn = self.rnn(out, h.detach())
        out = self.fc(out[-1, :, :])
        return out

