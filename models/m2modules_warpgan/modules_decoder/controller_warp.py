import torch
import torch.nn as nn
import torch.functional as tf


from warp_dense_image import sparse_image_warp


class WarpController(nn.Module):
    """
    
    Warp Controller network.
    
    """

    def __init__(self, in_channels: int, in_batch: int, in_height: int, in_width: int, n_ldmark: int = 16):
        """
        
        Warp Controller network.

        :param in_channels      : number of channels
        :param n_ldmark         : number of landmark points
        :param in_batch         : batch size
        :param in_height        : height of input image
        :param in_width         : width of input image
        :param initial          : initial channel number for convolution
        
        
        """

        super(WarpController, self).__init__()

        self.n_ldmark = n_ldmark
        self.n_height = in_height
        self.n_width  = in_width

        # inp: (in_batch, in_channels, in_height, in_width)
        # out: (in_batch, in_channels, in_height, in_width)
        self.flatten = nn.Flatten()

        in_features = in_channels * in_height * in_width

        # inp: (in_batch, in_channels * in_height * in_width)
        # out: (in_batch, 128)
        self.linear1 = nn.Linear(in_features=in_features, out_features=128)
    
        # inp: (in_batch, 128)
        # out: (in_batch, n_ldmark * 2)
        self.linear2 = nn.Linear(in_features=in_features, out_features=n_ldmark * 2)

        # inp: (in_batch, 128)
        # out: (in_batch, n_ldmark * 2)
        self.linear3 = nn.Linear(in_features=in_features, out_features=n_ldmark * 2)

        # initialize weights of layers
        self.initialize_weights()

        
    def forward(self, x: torch.Tensor, images_rendered: torch.Tensor, scales: torch.Tensor) -> tuple(torch.Tensor, torch.Tensor, torch.Tensor):
        """
        
        Forward function for Warp Controller.
        
        :param x : out of decoder controller
            :shape: (in_batch, in_channels, in_height, in_width)

        :param images_rendered : images rendered as output of warp decoder controller
            :shape: (in_batch, initial(default=64), in_height, in_width)

        :param scales   : scales values for input image
            :shape: (in_batch, 1)

        :return : out
            :shape: (batch_size, 2, 1, 4 * k)

        
        """

        # inp: (in_batch, in_channels, in_height, in_width)
        # out: (in_batch, in_channels, in_height, in_width)
        out = self.flatten(x)

        # inp: (in_batch, in_channels * in_height * in_width)
        # out: (in_batch, 128)
        out = self.linear1(out)

        # Control Points Prediction
        
        # shape: (1, self.n_ldmark * 2)
        landmarks_mean = torch.normal(mean=0, std=50, size=(self.n_ldmark, 2)) + \
                         torch.tensor([0.5 * self.n_height, 0.5 * self.n_width]).flatten().type(dtype=torch.float32)
        
        # inp: (in_batch, 128)
        # out: (in_batch, n_ldmark * 2)
        landmarks_pred = self.linear2(out)

        # shape: (in_batch, n_ldmark * 2)
        landmarks_pred += landmarks_mean

        # Displacements Prediction

        # inp: (in_batch, 128)
        # out: (in_batch, n_ldmark * 2)
        landmarks_displacement = self.linear3(out)

        # (in_batch, n_ldmark * 2) * (in_batch, 1)
        # out: (in_batch, n_ldmark * 2)
        landmarks_displacement *= self.scales

        # shape: (in_batch, n_ldmark, 2)
        landmarks_src = torch.reshape(landmarks_pred.detach().clone(), (-1, self.n_ldmark, 2))
        landmarks_dst = torch.reshape((landmarks_pred + landmarks_displacement).detach().clone(), (-1, self.n_ldmark, 2))

        # shape: (1)
        landmarks_norm = torch.mean(torch.norm(landmarks_src - landmarks_dst, dim=(1, 2)))

        # inp_images_rendered : (in_batch, initial(default=64), in_height, in_width)
        # inp_landmarks_src   : (in_batch, n_ldmark, 2)
        # inp_landmarks_dst   : (in_batch, n_ldmark, 2)
        
        # out_images_transformed: (in_batch, in_height, in_width, initial(default=64))
        # out_dense_flow        : (in_batch, in_height, in_width, 2)
        images_transformed, dense_flow = sparse_image_warp(images_rendered, landmarks_src, landmarks_dst, regularization_weight = 1e-6, num_boundary_points = 0)

        # reshape outputs to torch order
        # inp: (in_batch, in_height, in_width, initial(default=64))
        # out: (in_batch, initial(default=64), in_height, in_width)
        images_transformed = images_transformed.permute(0, 3, 1, 2)

        return images_transformed, landmarks_pred, landmarks_norm