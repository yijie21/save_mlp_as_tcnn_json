# save_torch_mlp_as_tcnn_json

This repository focuses on **Training a Multi-Layer Perceptron (MLP) using PyTorch and saving the weights in JSON format for tinycudann**. The motivation behind creating this repository stems from the observation that the tinycudann trainer lacks the efficacy of its PyTorch counterpart. In particular, the training loss in tinycudann may not decrease as rapidly as in PyTorch, and there is a risk of the training process failing to converge. To avoid such disadvantages of tinycudann but also leverage its lightning-fast inference speed, I created this repository which:

1. Trains an MLP in pytorch;
2. Saves the pytorch MLP weights as json files for tinycudann to load;
3. Loads json weights in tinycudann for fast inference.



## Pay attention to 

1. The input channels must be a multiple of 16, since the hardware matrix multipliers (TensorCores) operate on 16x16 matrix chunks. (refer to https://github.com/NVlabs/tiny-cuda-nn/issues/6 for more details)
2. The linear layers of tinycudann has no bias, so when training a pytorch MLP, set `bias=False` for `nn.Linear`
3. The meaning of `hidden_layers` is not the same for a pytorch MLP and a tinycudann MLP, usually, the number of hidden layers in a pytorch MLP equals to the number of hidden layers in a tinycudann MLP + 1



## Run an example

Simply run the `save_mlp_as_tcnn_json.py` file and you will see that the pytorch MLP and tinycudann MLP output nearly the same results.

