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


class Decoder(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = DoubleConv(in_channels, out_channels)
        # self.dropout = nn.Dropout2d(p=0.3)
    
    def forward(self, x, skip_features):

        x = self.conv_transpose(x)
        x = torch.cat([x, skip_features],dim=1)
        # x = self.dropout(x)
        x = self.conv_block(x)

        return x
        return x, conv_transpose


# In[10]:


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
    def __init__(self, n_input_channels, n_output_channels, n_features=64):
        super(UNet, self).__init__()
        
        self.down1 = Encoder(n_input_channels, n_features)
        self.down2 = Encoder(n_features, n_features*2)
        self.down3 = Encoder(n_features*2, n_features*4)
        self.down4 = Encoder(n_features*4, n_features*8)
        
        self.bridge = DoubleConv(n_features*8, n_features*16)
        
        self.up1 = Decoder(n_features*16, n_features*8)
        self.up2 = Decoder(n_features*8, n_features*4)
        self.up3 = Decoder(n_features*4, n_features*2)
        self.up4 = Decoder(n_features*2, n_features)
        
        self.outchannel1 = Branch1(n_features, n_features, n_output_channels)
        self.outchannel2 = Branch2(n_features, n_features, 3)
        
    def forward(self, x):
        
        conv1, pool1 = self.down1(x)
        conv2, pool2 = self.down2(pool1)
        conv3, pool3 = self.down3(pool2)
        conv4, pool4 = self.down4(pool3)
        
        bridge = self.bridge(pool4)
        
        decoder1 = self.up1(bridge, conv4)
        decoder2 = self.up2(decoder1, conv3)
        decoder3 = self.up3(decoder2, conv2)
        decoder4 = self.up4(decoder3, conv1)
        
        logits1 = self.outchannel1(decoder4)
        logits2 = self.outchannel2(decoder4)
        
        return logits1, logits2
    