import torch
from torch import nn
import torch.nn.functional as F

class LSTMDecoder(nn.Module):
    def __init__(self, hp, style_label=False):
        super(LSTMDecoder, self).__init__()
        self.hp = hp
        # to init hidden and cell from z:
        if style_label:
            self.fc_hc = nn.Linear(hp.Nz_dec + hp.style_dim, 2*hp.dec_hidden_size*hp.dec_layers)
            # LSTM params: input_size, hidden_size, num_layers
            self.lstm = nn.LSTM(hp.Nz_dec + hp.style_dim + hp.decoder_input_dim, hp.dec_hidden_size, hp.dec_layers, dropout=hp.dropout)
        else:
            self.fc_hc = nn.Linear(hp.Nz_dec, 2*hp.dec_hidden_size*hp.dec_layers)
            self.lstm = nn.LSTM(hp.Nz_dec + hp.decoder_input_dim, hp.dec_hidden_size, hp.dec_layers, dropout=hp.dropout)
        # create proba distribution parameters from hiddens:
        self.fc_params = nn.Linear(hp.dec_hidden_size, 6*hp.M + hp.pen_style_dim) # no pen state for now...

    def init_hc(self, z):
        hidden,cell = torch.split(torch.tanh(self.fc_hc(z)), self.hp.dec_hidden_size * self.hp.dec_layers, 1)
        if self.hp.dec_layers == 1:
            hidden_cell = (hidden.unsqueeze(0).contiguous(), cell.unsqueeze(0).contiguous())
        else:
            batch_size = hidden.size(0)
            hidden = hidden.reshape((batch_size, self.hp.dec_layers, -1)).permute(1, 0, 2).contiguous()
            cell = cell.reshape((batch_size, self.hp.dec_layers, -1)).permute(1, 0, 2).contiguous()
            hidden_cell = (hidden, cell)
        return hidden_cell


    def forward(self, inputs, z, labels=None, hidden_cell=None):
        if labels is not None:
            z = torch.cat([z, labels], dim=1)
        if hidden_cell is None:
            # then we must init from z
            hidden_cell = self.init_hc(z)

                #hidden = torch.zeros(self.hp.enc_layers*2, batch_size, self.hp.enc_hidden_size).cuda()
                #cell = torch.zeros(self.hp.enc_layers*2, batch_size, self.hp.enc_hidden_size).cuda()
        outputs,(hidden,cell) = self.lstm(inputs, hidden_cell)
        
        # output has shape (seq_len, batch, num_directions * hidden_size)

        #print ("decoder outputs", outputs.size())

        # in training we feed the lstm with the whole input in one shot
        # and use all outputs contained in 'outputs', while in generate
        # mode we just feed with the last generated sample:
        #if self.training:
        y = self.fc_params(outputs.view(-1, self.hp.dec_hidden_size))
        #else:
        #    y = self.fc_params(hidden.view(-1, self.hp.dec_hidden_size*self.hp.dec_layers))


        params = torch.split(y,6,1)
        #for i, item in enumerate(params):
        #    print (i, item.size())
        params_mixture = torch.stack(params[:-1]) # trajectory
        params_pen = params[-1] # additional pen states
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy = torch.split(params_mixture, 1, 2)

        #for item in [pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy]:
        #    print (item.size())

        if self.training:
            len_out = self.hp.Nmax + 1
        else:
            len_out = 1

        pi = F.softmax(pi.transpose(0,1).squeeze()).view(len_out,-1,self.hp.M)
        sigma_x = torch.exp(sigma_x.transpose(0,1).squeeze()).view(len_out,-1,self.hp.M)
        sigma_y = torch.exp(sigma_y.transpose(0,1).squeeze()).view(len_out,-1,self.hp.M)
        rho_xy = torch.tanh(rho_xy.transpose(0,1).squeeze()).view(len_out,-1,self.hp.M)
        mu_x = mu_x.transpose(0,1).squeeze().contiguous().view(len_out,-1,self.hp.M)
        mu_y = mu_y.transpose(0,1).squeeze().contiguous().view(len_out,-1,self.hp.M)
        pen_styles = F.softmax(params_pen).view(len_out, -1, self.hp.pen_style_dim)

        return pi,mu_x,mu_y,sigma_x,sigma_y,rho_xy,hidden,cell,pen_styles