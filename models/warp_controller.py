import torch
import torch.nn as nn
import torch.functional as tf
from torch.nn.modules.activation import ReLU

class WarpController(nn.Module):
    def __init__(self, batch_size, image_height, image_width, input_size_when_flatten, num_ldmark, scales):
        """
        
        Content Encoder network.
        :param batch_size                     : number of samples in a batch
        :param image_height                   : height of the input image
        :param image_width                    : width of the input image  
        :param input_size_when_flatten        : number of neurons in the first fully connected layer, equal to height * width * num_channels of the input
        :param num_ldmark                     : number of landmarks (?)
        :param scales                         : (?) shape (2 * num_ldmark , 1)
        
        """

        super().__init__()
        
        self.scales = scales
        self.num_ldmark = num_ldmark
        self.h = image_height
        self.w = image_width
        
        self.flatten = nn.Flatten()
        
        # inp: (batch_size, input_size_when_flatten)
        # out: (batch_size, 128)
        self.fc1 = nn.Linear(input_size_when_flatten, 128, bias = True)
        self.initialize_weights_with_he_biases_with_zero(self.fc1)
        
        # inp: (batch_size, 128)
        # out: (batch_size, 128)
        self.ln1 = nn.LayerNorm(128)
        
        # inp: (batch_size, 128)
        # out: (batch_size, 128)
        self.relu1 = nn.ReLU()

        # inp: (batch_size, 128)
        # out: (batch_size, num_ldmark * 2) 
        self.fc2 = nn.Linear(128, num_ldmark * 2, bias = False)
        nn.init.trunc_normal_(self.fc2.weight)
            
        # inp: (batch_size, num_ldmark * 2)
        # out: (batch_size, num_ldmark * 2) 
        self.fc3 = nn.Linear(128, num_ldmark * 2, bias = False)
        nn.init.trunc_normal_(self.fc3.weight)
        
            
    def initialize_weights_with_he_biases_with_zero(self, layer : nn.Module):
        nn.init.kaiming_normal_(layer.weight)
        layer.bias.data.fill_(0.0)
        
    def forward(self, x):
        """
        :param x: content encodings
        :shape: (in_batch, initial * 4, in_height/4, in_width/4)
        """

        # inp: (in_batch, initial * 4, in_height/4, in_width/4)
        # out: (batch_size, initial * 4 * in_heigt / 4 * in_width / 4) 
        x = self.flatten(x)
        # inp: (batch_size, initial * 4 * in_heigt / 4 * in_width / 4)
        # out: (in_batch, 128) 
        x = self.fc1(x)
        # inp: (in_batch, 128) 
        # out: (in_batch, 128) 
        x = self.ln1(x)
        # inp: (in_batch, 128) 
        # out: (in_batch, 128) 
        x = self.relu1(x)
            
       
        # ldmark_mean: shape (2 * num_ldmark, )
        ldmark_mean = (np.random.normal(0,50, (self.num_ldmark,2)) + np.array([[0.5*self.h,0.5*self.w]])).flatten()
        ldmark_mean = torch.tensor(ldmark_mean, dtype=torch.float32)

        # inp: (in_batch, 128) 
        # out: (in_batch, 2 * num_ldmark) 
        ldmark_pred = self.fc2(x)

        ldmark_pred = ldmark_pred + ldmark_mean

        # inp: (in_batch, 128) 
        # out: (in_batch, 2 * num_ldmark) 
        ldmark_diff = self.fc3(x)

        # ldmark_diff: shape (in_batch, 2 * num_ldmark)
        ldmark_diff = torch.reshape(self.scales, (-1, 1)) * ldmark_diff

        # src_pts: shape (in_batch, num_ldmark, 2)
        src_pts = torch.reshape(ldmark_pred, (-1, self.num_ldmark, 2))
        # dst_pts: shape (in_batch, num_ldmark, 2)
        dst_pts = torch.reshape(ldmark_pred + ldmark_diff, (-1, self.num_ldmark, 2))
        
        diff_norm = torch.mean(torch.norm(src_pts - dst_pts, dim = (1, 2)))
        # Şu son 3 satır???
        images_transformed, dense_flow = sparse_image_warp(warp_input, src_pts, dst_pts, regularization_weight = 1e-6, num_boundary_points=0)
        dense_flow = nn.Identity(dense_flow)
            
        # böyle multiple şey döndürmek okay mi
        return images_transformed, images_rendered, ldmark_pred, ldmark_diff
       