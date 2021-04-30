class HParams():
    def __init__(self):
        self.input_dim = 2
        
        # low DOF to high DOF
        self.encoder_input_dim = 6
        self.decoder_input_dim = 2
        self.pen_style_dim = 4
        
        self.style_dim = 3
        self.enc_layers = 1
        self.dec_layers = 1
        self.enc_hidden_size = 64  # 256
        self.dec_hidden_size = 128  # 512
        
        self.Nz = 2   # latent dimension
        self.Nz_dec = 4
        self.M = 3   # 20 # gaussian mixture
        self.dropout = 0.0  # 0.9
        
        #self.batch_size = 16
        self.eta_min = 0.01
        self.R = 0.99995
        self.KL_min = 0.2
        self.wKL = 0.5
        self.KL_a = 0.1
        self.KL_start = 0.01
        self.KL_delta = 0.01
        self.lr = 0.001
        self.lr_decay = 0.999
        self.min_lr = 0.00001
        self.grad_clip = 1.
        self.temperature = 0.4
        #self.max_seq_length = 200
        self.Nmax = 100
        self.save_every = 20