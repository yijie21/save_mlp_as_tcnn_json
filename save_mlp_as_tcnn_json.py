import torch
from torch import nn
import numpy as np
import tinycudann as tcnn
import json

### Pay Attention to:
# 1. The hidden layers of MLP and TCNN are different (MLP has one more hidden layer than TCNN)
# 2. TCNN mlp has no bias
# 3. Loading weights: https://github.com/NVlabs/tiny-cuda-nn/issues/6

class MLP(nn.Module):
    def __init__(self, in_dims, hidden_dims, out_dims, hidden_layers):
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
    def __init__(self, in_dims, hidden_dims, out_dims, hidden_layers):
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


def save_model_as_tcnn_json(mlp_model, save_path, hidden_dims, out_dims, dims_multiplier=16):
    mlp_model_keys = mlp_model.state_dict().keys()

    weights = []
    for key in mlp_model_keys:
        weights.append(mlp_model.state_dict()[key].cpu().numpy().flatten())

    # padding the last layer output dims to 32
    padded_dims = (out_dims // dims_multiplier + 1) * dims_multiplier
    weights[-1] = np.pad(weights[-1], (0, hidden_dims*padded_dims - weights[-1].flatten().shape[0]), 'constant', constant_values=1)

    # concatenate the weights
    weights = np.concatenate(weights)

    # save the weights to json
    json_weights = {}
    params_type = 'float'
    params_binary = {}
    
    bytes = weights.astype(np.float32).tobytes()
    params_binary['bytes'] = [byte for byte in bytes]
    params_binary['subtype'] = None
    
    json_weights['n_params'] = weights.shape[0]
    json_weights['params_type'] = params_type
    json_weights['params_binary'] = params_binary
    
    with open(save_path, 'w') as f:
        json.dump(json_weights, f)
    
        
def tcnn_read_json(json_path, tcnn_model):
    with open(json_path, 'r') as f:
        json_weights = json.load(f)
    
    params_binary = json_weights['params_binary']['bytes']
    params_fp32 = np.frombuffer(bytes(params_binary), dtype=np.float32)
    
    # load the weights to tcnn model
    checkpoint = {
        'net.params': torch.from_numpy(params_fp32)
    }
    
    # load the weights to tcnn model
    # pay attention that tcnn_model is in device or host
    tcnn_model.load_state_dict(checkpoint)
    
    return tcnn_model
    
in_dims = 32
hidden_dims = 128
out_dims = 75
hidden_layers = 3

save_path = "tcnn_mlp.json"

mlp_model = MLP(in_dims, hidden_dims, out_dims, hidden_layers)
tcnn_model = TCNN(in_dims, hidden_dims, out_dims, hidden_layers)

save_model_as_tcnn_json(mlp_model, save_path, hidden_dims, out_dims)
tcnn_read_json(save_path, tcnn_model)

# test the tcnn model
x = torch.randn(1, 32)
mlp_output = mlp_model(x)
tcnn_output = tcnn_model(x)

print("MLP output: ", mlp_output)
print("TCNN output: ", tcnn_output)