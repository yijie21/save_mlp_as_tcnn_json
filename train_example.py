import torch
from torch import nn
import tinycudann as tcnn
import numpy as np
import json
import imageio
import math
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from save_mlp_as_tcnn_json import save_model_as_tcnn_json, tcnn_read_json

# network
class MLP(nn.Module):
    def __init__(self, in_dims, hidden_dims, hidden_layers, out_dims):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dims, hidden_dims, bias=False),
            nn.ReLU(inplace=True),
            *[nn.Sequential(
                nn.Linear(hidden_dims, hidden_dims, bias=False),
                nn.ReLU(inplace=True)
            ) for _ in range(hidden_layers)],
            nn.Linear(hidden_dims, out_dims, bias=False)
        )
        
        # randomly init the weights and bias
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                # nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.net(x)

class TCNN(nn.Module):
    def __init__(self, in_dims, hidden_dims, hidden_layers, out_dims):
        super().__init__()
        
        self.net = tcnn.Network(
            n_input_dims=in_dims,
            n_output_dims=out_dims,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dims,
                "n_hidden_layers": hidden_layers + 1,
            },
        )
    
    def forward(self, x):
        return self.net(x)

# dataset
class ImageSample(Dataset):
    def __init__(self, image_path, batch_size, iters, position_dims):
        super().__init__()
        self.image_path = image_path
        self.input_data = None
        self.h = 0
        self.w = 0
        self.batch_size = batch_size
        self.iters = iters
        self.position_dims = position_dims
        self.read_image()
    
    def format_coords(self):
        x = torch.arange(self.w).float() / self.w
        y = torch.arange(self.h).float() / self.h
        grid_x, grid_y = torch.meshgrid(y, x)
        coords = torch.stack([grid_y, grid_x], dim=-1)
        return coords
    
    def position_encoding(self):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if self.position_dims % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dimension (got dim={:d})".format(self.position_dims))
        pe = torch.zeros(self.position_dims, self.h, self.w)
        # Each dimension use half of d_model
        position_dims_half = int(self.position_dims / 2)
        div_term = torch.exp(torch.arange(0., position_dims_half, 2) *
                            -(math.log(10000.0) / position_dims_half))
        pos_w = torch.arange(0., self.w).unsqueeze(1)
        pos_h = torch.arange(0., self.h).unsqueeze(1)
        pe[0:position_dims_half:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, self.h, 1)
        pe[1:position_dims_half:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, self.h, 1)
        pe[position_dims_half::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, self.w)
        pe[position_dims_half + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, self.w)

        return pe.permute(1, 2, 0)
    
    def read_image(self):
        img_hwc = torch.from_numpy(imageio.imread(self.image_path)) / 255.
        self.h = img_hwc.shape[0]
        self.w = img_hwc.shape[1]
        coords_hw2 = self.format_coords()
        pe_hwc = self.position_encoding()
        self.input_data = torch.cat([img_hwc, pe_hwc], dim=-1).reshape(-1, 3 + self.position_dims)
        
    def __len__(self):
        return self.h * self.w
    
    def __getitem__(self, idx):
        return self.input_data[idx]

# inference function
def inference(network, dataset, save_path):
    coords = dataset.input_data.cuda()[:, 3:]
    h = dataset.h
    w = dataset.w
    
    # infer 
    with torch.no_grad():
        pred = network(coords)
    
    # save inference result    
    pred = torch.clamp(pred.reshape(h, w, 3), 0)
    pred_np = (pred.cpu().numpy() * 255).astype(np.uint8)
    imageio.imwrite(save_path, pred_np)

# training preparations
position_dims = 32
in_dims = position_dims
hidden_dims = 64
hidden_layers = 4
out_dims = 3
batch_size = 4096
iters = 1000
epochs = 10

network = MLP(in_dims, hidden_dims, hidden_layers, out_dims).cuda().train()
tcnn_net = TCNN(in_dims, hidden_dims, hidden_layers, out_dims).cuda()
optim = torch.optim.Adam(network.parameters(), lr=1e-4)

# dataLoader
picture_path = 'D:/code/CodeUtils/save_mlp_as_tcnn_json/picture.jpg'
save_path = 'D:/code/CodeUtils/save_mlp_as_tcnn_json/save.jpg'
dataset = ImageSample(picture_path, batch_size, iters, position_dims)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# training
for epoch in range(epochs):
    for i, batch in enumerate(tqdm(dataloader)):
        batch = batch.cuda()
        coords = batch[:, 3:]
        colors = batch[:, :3]
        
        pred = network(coords)
        
        loss = torch.nn.L1Loss()(pred, colors)
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        if i % 100 == 0:
            print(f"epoch {epoch} steps {i}, loss {loss.item()}")
            
    inference(network, dataset, save_path)
    
# # compare results of tcnn and torch
save_json_path = 'D:/code/CodeUtils/save_mlp_as_tcnn_json/weights.json'
tcnn_save_path = 'D:/code/CodeUtils/save_mlp_as_tcnn_json/save_tcnn.jpg'
save_model_as_tcnn_json(network, save_json_path, hidden_dims, out_dims)
tcnn_read_json(save_json_path, tcnn_net)

inference(tcnn_net, dataset, tcnn_save_path)