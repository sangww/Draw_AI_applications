from LSTMDecoder import LSTMDecoder
import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from torch import nn

def lr_decay(hp, optimizer):
    """Decay learning rate by a factor of lr_decay"""
    for param_group in optimizer.param_groups:
        if param_group['lr']>hp.min_lr:
            param_group['lr'] *= hp.lr_decay
    return optimizer

def sample_bivariate_normal(hp, mu_x,mu_y,sigma_x,sigma_y,rho_xy, greedy=False):
    # inputs must be floats
    if greedy:
      return mu_x,mu_y
    mean = [mu_x, mu_y]
    sigma_x *= np.sqrt(hp.temperature)
    sigma_y *= np.sqrt(hp.temperature)
    cov = [[sigma_x * sigma_x, rho_xy * sigma_x * sigma_y],\
        [rho_xy * sigma_x * sigma_y, sigma_y * sigma_y]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]

MSELoss = nn.MSELoss()

class Decoder(nn.Module):
    def __init__(self, hp):
        super(Decoder, self).__init__()
        self.hp = hp
        self.lstm = nn.LSTM(hp.decoder_input_dim, hp.dec_hidden_size, hp.dec_layers, dropout=hp.dropout)
        self.fc_params = nn.Linear(hp.dec_hidden_size, 6*hp.M + hp.pen_style_dim)

    def init_hc(self, batch_size):
        hidden = torch.zeros(self.hp.dec_layers, batch_size, self.hp.dec_hidden_size).cuda()
        cell = torch.zeros(self.hp.dec_layers, batch_size, self.hp.dec_hidden_size).cuda()
        return hidden, cell

    def forward(self, inputs, hidden_cell=None):
        batch_size = inputs.size(1)
        if hidden_cell is None:
            # then we must init from z
            hidden_cell = self.init_hc(batch_size)
        outputs,(hidden,cell) = self.lstm(inputs, hidden_cell)
        
        # output has shape (seq_len, batch, num_directions * hidden_size)

        #print ("decoder outputs", outputs.size())

        # in training we feed the lstm with the whole input in one shot
        # and use all outputs contained in 'outputs', while in generate
        # mode we just feed with the last generated sample:
        #if self.training:
        y = self.fc_params(outputs.view(-1, self.hp.dec_hidden_size))


        params = torch.split(y,6,1)
        #for i, item in enumerate(params):
        #    print (i, item.size())
        params_mixture = torch.stack(params[:-1]) # trajectory
        params_pen = params[-1] # additional pen states
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy = torch.split(params_mixture, 1, 2)

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

class UnconditionalDecoder():
    def __init__(self, hp):
        self.hp = hp
        self.decoder = Decoder(hp).cuda()

        self.optim_decoder = optim.Adam(self.decoder.parameters(), hp.lr)
        self.iters = 0
        self.echo_every = 20

    def bivariate_normal_pdf(self, dx, dy):
        z_x = ((dx-self.mu_x)/self.sigma_x)**2
        z_y = ((dy-self.mu_y)/self.sigma_y)**2
        z_xy = (dx-self.mu_x)*(dy-self.mu_y)/(self.sigma_x*self.sigma_y)
        z = z_x + z_y -2*self.rho_xy*z_xy
        exp = torch.exp(-z/(2*(1-self.rho_xy**2)))
        norm = 2*np.pi*self.sigma_x*self.sigma_y*torch.sqrt(1-self.rho_xy**2)
        return exp / norm

    def reconstruction_loss(self, mask, dx, dy, pen_styles_target, batch_size):
        pdf = self.bivariate_normal_pdf(dx, dy)
        # position difference
        L1 = -torch.sum(mask * torch.log(1e-5 + torch.sum(self.pi * pdf, 2)))\
            / float(self.hp.Nmax * batch_size)
        
        # additional pen styles difference
        pen_styles_target = pen_styles_target.permute(1, 0, 2).contiguous().view(batch_size, -1)
        self.pen_styles = self.pen_styles.permute(1, 0, 2).contiguous().view(batch_size, -1)
        # print(pen_styles_target.size(), self.pen_styles.size())
        L2 = MSELoss(pen_styles_target, self.pen_styles)
        #LP = -torch.sum(pen_styles_target*torch.log(self.pen_styles))/float(self.hp.Nmax*batch_size)
        if self.iters % self.echo_every == 0:
          print("position and styles loss =", L1.item(), L2.item())
        return L1+L2 

    # assume equal lengths
    def make_target(self, batch):
        batch_size = batch.size(1)
        eos = torch.stack([torch.Tensor([0.0] * 5)] * batch_size).cuda().unsqueeze(0)
        #print(batch.size(), eos.size())
        batch = torch.cat([batch, eos], 0)

        mask = torch.ones(self.hp.Nmax + 1, batch_size)
        mask[self.hp.Nmax, :] = 0.0
        mask = mask.cuda()

        dx = torch.stack([batch.data[:, :, 0]] * self.hp.M, 2)
        dy = torch.stack([batch.data[:, :, 1]] * self.hp.M, 2)
        
        # additional pen dimensions
        pen_states = []
        for i in range(self.hp.pen_style_dim):
            pen_states.append(batch.data[:, :, 2+i])
        p = torch.stack(pen_states,2)

        return mask, dx, dy, p

    def train(self, dataloader, epoch):
        self.iters += 1
        self.decoder.train()

        for i, data in enumerate(dataloader):
            inputs = Variable(data).cuda()
            batch_size = inputs.size(0)

            # N C L -> L N C
            inputs = inputs.permute(2, 0, 1)

            assert batch_size == inputs.size(1)
            assert self.hp.Nmax == inputs.size(0)

            sos = torch.stack([torch.Tensor([0.0] * self.hp.decoder_input_dim)] * batch_size).cuda().unsqueeze(0)

            stroke = inputs[:, :, :2] # x,y only
            decoder_inputs = torch.cat([sos, stroke], 0)

            self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, self.rho_xy, \
                hidden, cell, self.pen_styles = self.decoder(decoder_inputs)

            mask, dx, dy, pen_styles_target = self.make_target(inputs)
            
            self.optim_decoder.zero_grad()

            loss = self.reconstruction_loss(mask, dx, dy, pen_styles_target, batch_size)
            loss.backward()

            nn.utils.clip_grad_norm_(self.decoder.parameters(), self.hp.grad_clip)

            self.optim_decoder.step()

        self.optim_decoder = lr_decay(self.hp, self.optim_decoder)

        #if epoch > 0 and epoch % self.hp.save_every == 0:
        #    self.save(epoch)

    def generate_with_2d(self, inputs):
        self.decoder.train(False)

        # L N C
        batch_size = inputs.size(1)
        assert batch_size == 1

        sos = torch.Tensor([0.0, 0.0]).cuda().view(1,1,2)

        decoder_inputs = torch.cat([sos, inputs], 0)
        self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, \
                self.rho_xy, hidden, cell, pen_styles = \
                    self.decoder(decoder_inputs)

        dx = self.mu_x.squeeze().cpu().detach().numpy()[:,1]
        dy = self.mu_y.squeeze().cpu().detach().numpy()[:,1]
        pen_styles = pen_styles.squeeze().cpu().detach().numpy()
        x_sample = np.cumsum(dx, 0)
        y_sample = np.cumsum(dy, 0)
        return x_sample, y_sample, dx, dy, pen_styles
      
    