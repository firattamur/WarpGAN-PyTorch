class WarpController(nn.Module):
    def __init__(self, batch_size, input_size_when_flatten, num_ldmark, scales):
         
        super().__init__()
        
        self.scales = scales
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(input_size_when_flatten, 128, bias = True)
        self.initialize_weights_with_he_biases_with_zero(self.fc1)
        
        self.ln1 = nn.LayerNorm(128)
        
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(128, num_ldmark * 2, bias = False)
        self.init.trunc_normal_(self.fc2.weights)
        self.fc2.bias.fill_(0.0)
        
        self.fc3 = nn.Linear(num_ldmark * 2, num_ldmark * 2, bias = False)
        self.init.trunc_normal_(self.fc3.weights)
        self.fc3.bias.fill_(0.0)
      
        
    def initialize_weights_with_he_biases_with_zero(self, layer : nn.Module):
        nn.init.kaiming_normal_(layer.weight)
        layer.bias.data.fill_(0.0)
    
    def forward(self, x):
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.relu1(x)
        
        # BÖYLE BİR CONSTANT OLUŞTURMAK BURAYI BOZAR MI
        ldmark_mean = (np.random.normal(0,50, (num_ldmark,2)) + np.array([[0.5*h,0.5*w]])).flatten()
        ldmark_mean = torch.tensor(ldmark_mean, dtype=torch.float32)

        ldmark_pred = self.fc2(x)
        ldmark_pred = ldmark_pred + ldmark_mean
        #tf.identity ile aynı mı?
        ldmark_pred = nn.Identity(ldmark_pred)
        ldmark_diff = self.fc3(x)
        ldmark_diff = nn.Identity(ldmark_diff)
        # scales i yukarda bu şekilde dahil etmek problem mi
        ldmark_diff = nn.Identity(torch.reshape(scales, (-1, 1)) * ldmark_diff)
        
        src_pts = torch.reshape(ldmark_pred, (-1, num_ldmark, 2))
        dst_pts = torch.reshape(ldmark_pred + ldmark_diff, (-1, num_ldmark, 2))
        
        diff_norm = torch.mean(torch.norm(src_pts - dst_pts, dim = (1, 2)))
        images_transformed, dense_flow = sparse_image_warp(warp_input, src_pts, dst_pts, regularization_weight = 1e-6, num_boundary_points=0)
        dense_flow = nn.Identity(dense_flow)
        
        # böyle multiple şey döndürmek okay mi
        return images_transformed, images_rendered, ldmark_pred, ldmark_diff
        