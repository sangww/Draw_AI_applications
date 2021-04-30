import torch
from torch import nn
import torch.nn.functional as F

class BiLSTMEncoder(nn.Module):
    def __init__(self, hp, style_label=False):
        super(BiLSTMEncoder, self).__init__()
        self.hp = hp
        self.lstm = nn.LSTM(hp.encoder_input_dim, hp.enc_hidden_size, hp.enc_layers, dropout=hp.dropout, bidirectional=True)
        if style_label:
            self.fc_mu = nn.Linear(2*hp.enc_layers*hp.enc_hidden_size + hp.style_dim, hp.Nz)
            self.fc_sigma = nn.Linear(2*hp.enc_layers*hp.enc_hidden_size + hp.style_dim, hp.Nz)
        else:
            self.fc_mu = nn.Linear(2*hp.enc_layers*hp.enc_hidden_size, hp.Nz)
            self.fc_sigma = nn.Linear(2*hp.enc_layers*hp.enc_hidden_size, hp.Nz)

        self.train()

    def forward(self, inputs, labels=None, hidden_cell=None):
        batch_size = inputs.size(1) # input is L N C
        if hidden_cell is None:
            hidden = torch.zeros(self.hp.enc_layers*2, batch_size, self.hp.enc_hidden_size).cuda()
            cell = torch.zeros(self.hp.enc_layers*2, batch_size, self.hp.enc_hidden_size).cuda()
            hidden_cell = (hidden, cell)
        _, (hidden, cell) = self.lstm(inputs, hidden_cell)
        # hidden is (2, batch_size, hidden_size), we want (batch_size, 2*hidden_size):
        hidden_forward, hidden_backward = torch.split(hidden,self.hp.enc_layers,0)
        hidden_forward = hidden_forward.permute(1, 0, 2).reshape((batch_size, -1))
        hidden_backward = hidden_backward.permute(1, 0, 2).reshape((batch_size, -1))
        if labels is None:
            hidden_cat = torch.cat([hidden_forward, hidden_backward],1)
        else:
            hidden_cat = torch.cat([hidden_forward, hidden_backward, labels],1)

        mu = self.fc_mu(hidden_cat)
        sigma_hat = self.fc_sigma(hidden_cat)
        sigma = torch.exp(sigma_hat / 2.0)

        z_size = mu.size()
        N = torch.normal(torch.zeros(z_size), torch.ones(z_size)).cuda()
        z = mu + sigma * N

        return z, mu, sigma_hat