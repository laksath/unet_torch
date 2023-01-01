import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class Encoder(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.max = nn.MaxPool2d(2)
        self.conv_block = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        
        conv = self.conv_block(x)
        pool = self.max(conv)

        return conv, pool

class DecoderB1(nn.Module):
    
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = DoubleConv(mid_channels, out_channels)
    
    def forward(self, x, skip_features, skip_features_b2):

        x = self.conv_transpose(x)
        x = torch.cat([x, skip_features, skip_features_b2],dim=1)
        x = self.conv_block(x)

        return x

class DecoderB2(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = DoubleConv(in_channels, out_channels)
    
    def forward(self, x, skip_features):

        conv_transpose = self.conv_transpose(x)
        x = torch.cat([conv_transpose, skip_features],dim=1)
        x = self.conv_block(x)

        return x, conv_transpose

class Branch1(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        
        self.conv_block = DoubleConv(in_channels, mid_channels)
        self.conv_2d = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv_block(x)
        x = self.conv_2d(x)
        x = self.sigmoid(x)
        
        return x

class Branch2(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        
        self.conv_block = DoubleConv(in_channels, mid_channels)
        self.conv_2d = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv_block(x)
        x = self.conv_2d(x)
        x = self.sigmoid(x)
        
        return x

class UNet(nn.Module):
    def __init__(self, input_shape=256, n_input_channels=3, n_output_channels_b1=1, n_output_channels_b2=3, n_features=64, latent_dim=128):
        super(UNet, self).__init__()
        self.input_shape = input_shape
        self.n_features = n_features
        
        self.down1 = Encoder(n_input_channels, n_features)
        self.down2 = Encoder(n_features, n_features*2)
        self.down3 = Encoder(n_features*2, n_features*4)
        self.down4 = Encoder(n_features*4, n_features*8)
        
        self.bridge = DoubleConv(n_features*8, n_features*16)
        
        self.upb1_1 = DecoderB1(n_features*16,n_features*24, n_features*8)
        self.upb1_2 = DecoderB1(n_features*8,n_features*12, n_features*4)
        self.upb1_3 = DecoderB1(n_features*4,n_features*6, n_features*2)
        self.upb1_4 = DecoderB1(n_features*2,n_features*3, n_features)
        
        flatten_size = n_features*16 * int(input_shape/16) * int(input_shape/16)
        
        self.intermediate1 = nn.Sequential(
            nn.Linear(flatten_size, latent_dim*2),
            nn.BatchNorm1d(latent_dim*2),
            nn.ReLU(inplace=True),
        )
        
        self.fc_mu = nn.Linear(latent_dim*2, latent_dim)
        self.fc_var = nn.Linear(latent_dim*2, latent_dim)
        
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, flatten_size),
            nn.ReLU(inplace=True),
        )
        
        self.upb2_1 = DecoderB2(n_features*16, n_features*8)
        self.upb2_2 = DecoderB2(n_features*8, n_features*4)
        self.upb2_3 = DecoderB2(n_features*4, n_features*2)
        self.upb2_4 = DecoderB2(n_features*2, n_features)
        
        self.outchannel_b1_grayscale = Branch1(n_features, n_features, n_output_channels_b1)
        self.outchannel_b1_rgb = Branch2(n_features, n_features, n_output_channels_b2)
        self.outchannel_b2_grayscale = Branch2(n_features, n_features, n_output_channels_b2)
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        
        return eps * std + mu
    
    def forward(self, x):
        
        conv1, pool1 = self.down1(x)
        conv2, pool2 = self.down2(pool1)
        conv3, pool3 = self.down3(pool2)
        conv4, pool4 = self.down4(pool3)
        
        bridge = self.bridge(pool4)
        
        #2nd branch vae
        
        flattened_bridge = torch.flatten(bridge, start_dim=1)
        intermediate1 = self.intermediate1(flattened_bridge)
        
        mu = self.fc_mu(intermediate1)
        log_var = self.fc_var(intermediate1)
        
        z = self.reparameterize(mu,log_var)
        
        result = self.decoder_input(z)
        reshaped_result = result.view(-1, self.n_features*16, int(self.input_shape/16),int(self.input_shape/16))
        
        # 2nd branch
        decoder_b2_1,decoder_b2_conv_transpose_1 = self.upb2_1(reshaped_result, conv4)
        decoder_b2_2,decoder_b2_conv_transpose_2 = self.upb2_2(decoder_b2_1, conv3)
        decoder_b2_3,decoder_b2_conv_transpose_3 = self.upb2_3(decoder_b2_2, conv2)
        decoder_b2_4,decoder_b2_conv_transpose_4 = self.upb2_4(decoder_b2_3, conv1)
                
        b2_rgb = self.outchannel_b2_grayscale(decoder_b2_4)
        
        # 1st branch
        decoder_b1_1 = self.upb1_1(bridge, decoder_b2_conv_transpose_1, conv4)
        decoder_b1_2 = self.upb1_2(decoder_b1_1, decoder_b2_conv_transpose_2, conv3)
        decoder_b1_3 = self.upb1_3(decoder_b1_2, decoder_b2_conv_transpose_3, conv2)
        decoder_b1_4 = self.upb1_4(decoder_b1_3, decoder_b2_conv_transpose_4, conv1)
                
        b1_gry = self.outchannel_b1_grayscale(decoder_b1_4)
        b1_rgb = self.outchannel_b1_rgb(decoder_b1_4)
        
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        
        return b1_gry, b1_rgb, b2_rgb, kld_loss