from BiLSTMEncoder import BiLSTMEncoder
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

class SketchTransfer_nolabel():
    def __init__(self, hp):
        self.hp = hp
        self.encoder = BiLSTMEncoder(hp, False).cuda()
        self.decoder = LSTMDecoder(hp, False).cuda()

        self.optim_encoder = optim.Adam(self.encoder.parameters(), hp.lr)
        self.optim_decoder = optim.Adam(self.decoder.parameters(), hp.lr)
        self.KL_weight = hp.KL_start
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
        L2 = MSELoss(pen_styles_target, self.pen_styles) * 10
        #LP = -torch.sum(pen_styles_target*torch.log(self.pen_styles))/float(self.hp.Nmax*batch_size)
        if self.iters % self.echo_every == 0:
          print("position and styles loss =", L1.item(), L2.item())
        return L2 #L1+L2 

    def KL_loss(self, batch_size):
        LKL = -0.5*torch.sum(1+self.sigma_stroke - self.mu_stroke**2-torch.exp(self.sigma_stroke))\
            / float(self.hp.Nz * batch_size)
        KL_min = Variable(torch.Tensor([self.hp.KL_min]).cuda()).detach()
        #return hp.wKL*self.eta_step * torch.max(LKL,KL_min)
        KL_loss = self.hp.wKL * self.KL_weight * torch.max(LKL,KL_min)
        if self.iters % self.echo_every == 0:
          print("KL loss =", KL_loss.item())
        return KL_loss

    # assume equal lengths
    def make_target(self, batch):
        batch_size = batch.size(1)
        eos = torch.stack([torch.Tensor([0.0] * 6)] * batch_size).cuda().unsqueeze(0)
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
        self.encoder.train()
        self.decoder.train()

        for i, data in enumerate(dataloader):
            inputs = Variable(data).cuda()
            batch_size = inputs.size(0)

            # N C L -> L N C
            inputs = inputs.permute(2, 0, 1)

            assert batch_size == inputs.size(1)
            assert self.hp.Nmax == inputs.size(0)

            z, self.mu_stroke, self.sigma_stroke = self.encoder(inputs, None)

            sos = torch.stack([torch.Tensor([0.0] * self.hp.decoder_input_dim)] * batch_size).cuda().unsqueeze(0)

            stroke = inputs[:, :, :2] # x,y only
            decoder_inputs = torch.cat([sos, stroke], 0)

            z_stack = torch.stack([z] * (self.hp.Nmax+1))

            # decoder concatenates sequence and z at every time step
            decoder_inputs = torch.cat([decoder_inputs, z_stack], 2)

            self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, self.rho_xy, \
                hidden, cell, self.pen_styles = self.decoder(decoder_inputs, z)

            mask, dx, dy, pen_styles_target = self.make_target(inputs)
            

            self.optim_encoder.zero_grad()
            self.optim_decoder.zero_grad()

            L_KL = self.KL_loss(batch_size)
            L_R = self.reconstruction_loss(mask, dx, dy, pen_styles_target, batch_size)
            loss = L_KL + L_R
            loss.backward()

            nn.utils.clip_grad_norm_(self.encoder.parameters(), self.hp.grad_clip)
            nn.utils.clip_grad_norm_(self.decoder.parameters(), self.hp.grad_clip)

            self.optim_encoder.step()
            self.optim_decoder.step()

        #print("Epoch", epoch, "Loss KL", L_KL.item(), "Loss R", L_R.item())
        self.optim_encoder = lr_decay(self.hp, self.optim_encoder)
        self.optim_decoder = lr_decay(self.hp, self.optim_decoder)
        
        if self.KL_weight < 1.0:
            self.KL_weight += self.hp.KL_delta

        if epoch > 0 and epoch % self.hp.save_every == 0:
            self.save(epoch)

    def generate_with_2d(self, inputs, latent, greedy=False):
        self.decoder.train(False)

        # L N C
        batch_size = inputs.size(1)
        assert batch_size == 1

        z = torch.Tensor(latent).view(1, self.hp.Nz).cuda()

        sos = Variable(torch.Tensor([0.0, 0.0]).view(1,1,-1).cuda())
        s = sos
        seq_x = []
        seq_y = []
        seq_z = []
        hidden_cell = None
        for i in range(self.hp.Nmax):
            decoder_inputs = torch.cat([s, z.unsqueeze(0)], 2)

            # decode:
            self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, \
                self.rho_xy, hidden, cell, pen_styles = \
                    self.decoder(decoder_inputs, z, None, hidden_cell)
            hidden_cell = (hidden, cell)
            # sample from parameters:
            #s, dx, dy, pen_down, eos = self.sample_next_state()
            s, dx, dy = self.sample_next_state(greedy)
            #------
            seq_x.append(dx)
            seq_y.append(dy)
            seq_z.append(pen_styles[0,0,:].cpu().detach().numpy())
        # visualize result:

        x_sample = np.cumsum(seq_x, 0)
        y_sample = np.cumsum(seq_y, 0)
        seq_z = np.array(seq_z)
        return x_sample, y_sample, seq_x, seq_y, seq_z



    def sample_next_state(self, greedy=False):

        def adjust_temp(pi_pdf):
            pi_pdf = np.log(pi_pdf) / self.hp.temperature
            pi_pdf -= pi_pdf.max()
            pi_pdf = np.exp(pi_pdf)
            pi_pdf /= pi_pdf.sum()
            return pi_pdf

        # get mixture indice:
        pi = self.pi.data[0,0,:].cpu().numpy()
        pi = adjust_temp(pi)
        pi_idx = np.random.choice(self.hp.M, p=pi)
        #print (pi_idx)
        # get pen state:
        #q = self.q.data[0,0,:].cpu().numpy()
        #q = adjust_temp(q)
        #q_idx = np.random.choice(3, p=q)
        # get mixture params:
        mu_x = self.mu_x.data[0,0,pi_idx].item()
        mu_y = self.mu_y.data[0,0,pi_idx].item()
        sigma_x = self.sigma_x.data[0,0,pi_idx].item()
        sigma_y = self.sigma_y.data[0,0,pi_idx].item()
        rho_xy = self.rho_xy.data[0,0,pi_idx].item()
        x,y = sample_bivariate_normal(self.hp, mu_x,mu_y,sigma_x,sigma_y,rho_xy,greedy=greedy)
        next_state = torch.zeros(2) ## temp
        next_state[0] = x
        next_state[1] = y
        #next_state[q_idx+2] = 1
        return Variable(next_state.cuda()).view(1,1,-1), x, y

    def save(self, epoch):
        sel = np.random.rand()
        torch.save(self.encoder.state_dict(), \
            'sketch_encoder_sel_%3f_epoch_%d.pth' % (sel,epoch))
        torch.save(self.decoder.state_dict(), \
            'sketch_decoder_sel_%3f_epoch_%d.pth' % (sel,epoch))