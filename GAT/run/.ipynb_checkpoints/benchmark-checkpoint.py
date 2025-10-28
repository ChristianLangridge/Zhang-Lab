#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import torch
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_line, theme_bw, theme, geom_point, geom_hline, annotate, scale_y_log10
import plotnine as p9


# In[2]:





# In[3]:


# Dynamically set the absolute path to the LEMBAS directory

# Replace the current path setup with this more robust version
import os
import sys

# Get the absolute path to the project root directory (parent of 'run')
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the project root to Python path so LEMBAS can be found
sys.path.append(project_root)

# Now try importing LEMBAS
from LEMBAS.model.bionetwork import format_network, SignalingModel
from LEMBAS.model.train import train_signaling_model
import LEMBAS.utilities as utils
from LEMBAS import plotting, io


# In[ ]:


def benchmark_this_dict(dict_to_bench):
    # 设置多核计算
    n_cores = 12
    utils.set_cores(n_cores)

    # 设置随机种子
    seed = 888
    if seed:
        torch.use_deterministic_algorithms(True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        utils.set_seeds(seed=seed)

    device = dict_to_bench["device"]
    data_path = '../data'
    print(f"Current device is set to {device}")

    # 打印字典中值为1的测试项
    keys_with_one = [key for key, value in dict_to_bench.items() if value == 1]
    print(f"Tests to run: {keys_with_one}")

    # 加载数据（使用绝对路径）
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
    net = pd.read_csv(os.path.join(data_path, 'network.tsv'), sep='\t', index_col=False)
    ligand_input = pd.read_csv(os.path.join(data_path, 'TF.tsv'), sep='\t', low_memory=False, index_col=0)
    tf_output = pd.read_csv(os.path.join(data_path, 'Geneexpression.tsv'), sep='\t', low_memory=False, index_col=0)

    # 格式化网络
    net = format_network(net, weight_label='Interaction')

    # 定义网络和训练的超参数
    projection_amplitude_in = 1.2
    projection_amplitude_out = 1.2
    bionet_params = {'target_steps': 150, 'max_steps': 10, 'exp_factor': 50, 'tolerance': 1e-20, 'leak': 1e-2}
    lr_params = {'max_iter': 2, 'learning_rate': 2e-4}
    other_params = {'batch_size': 20, 'noise_level': 10, 'gradient_noise_level': 1e-9}
    regularization_params = {
        'param_lambda_L2': 1e-6, 
        'moa_lambda_L1': 0.1, 
        'ligand_lambda_L2': 1e-5, 
        'uniform_lambda_L2': 1e-4, 
        'uniform_max': 1/projection_amplitude_out, 
        'spectral_loss_factor': 1e-5
    }
    spectral_radius_params = {'n_probes_spectral': 5, 'power_steps_spectral': 10, 'subset_n_spectral': 10}
    hyper_params = {**lr_params, **other_params, **regularization_params, **spectral_radius_params}

    # 定义模型
    mod = SignalingModel(
        net=net,
        X_in=ligand_input,
        y_out=tf_output,
        projection_amplitude_in=projection_amplitude_in,
        projection_amplitude_out=projection_amplitude_out,
        weight_label='Interaction',
        source_label='TF',
        target_label='Gene',
        bionet_params=bionet_params,
        dtype=torch.float32,
        device=device,
        seed=seed
    )

    # 打印模型参数名称
    print("Model state_dict parameters:")
    for name, param in mod.state_dict().items():
        param_sum = param.sum().item()  # 计算张量所有元素的和
        print(f"{name}: {param} Sum: {param_sum}")

    # 将数据转换为张量
    mod.X_in = mod.df_to_tensor(mod.X_in)
    mod.y_out = mod.df_to_tensor(mod.y_out)
 #   print('mod.X_in', mod.X_in.shape, mod.y_out.shape)
    # 模型设置
    mod.input_layer.weights.requires_grad = False  # 不学习输入的缩放因子
   # mod.signaling_network.prescale_weights(target_radius=0.8)  # 调整谱半径

    # 定义损失函数和优化器
    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam

    print("Training has started")
    print("-----------------------------------------------------")
    print("Printing training progress")
    print("-----------------------------------------------------")

    # 训练模型
    mod, cur_loss, cur_eig, mean_loss, stats, X_train, X_test, X_val, y_train, y_test, y_val, train_dataloader = train_signaling_model(
        mod, optimizer, loss_fn,
        reset_epoch=200,
        hyper_params=hyper_params,
        train_seed=seed,
        verbose=True,
        dict_to_bench=dict_to_bench
    )

    # 模型评估前切换到评估模式
    mod.eval()
    print("Model state_dict parameters:")
    for name, param in mod.state_dict().items():
        param_sum = param.sum().item()  # 计算张量所有元素的和
        print(f"{name}: {param} Sum: {param_sum}")
    # 在训练完成后，保存以下权重到文件：input_layer.weights, signaling_network.weights, output_layer.weights
    state_dict = mod.state_dict()
    weight_keys = ["input_layer.weights", "signaling_network.weights", "output_layer.weights"]
    for key in weight_keys:
        if key in state_dict:
            tensor = state_dict[key]
            # 将点号替换为下划线，作为文件名的一部分
            file_name = "/home/x_kejli" + key.replace('.', '_') + ".pt"
            torch.save(tensor, file_name)
            print(f"Saved {key} to {file_name}")
        else:
            print(f"Warning: {key} not found in model.state_dict()!")

    # 计算预测结果和皮尔逊相关系数
   # print('mod.X_in', mod.X_in.shape)

    last_result = []
    for batch, (X_in_, y_out_) in enumerate(train_dataloader):
        #optimizer.zero_grad()

        X_in_, y_out_ = X_in_.to(mod.device), y_out_.to(mod.device)
        Y_hat, Y_full = mod(X_in_)
        last_result.append(Y_hat)

    last_result = torch.cat(last_result, dim=0)
    Y_hat = last_result    



    pr, _ = pearsonr(mod.y_out.detach().flatten().cpu().numpy(), Y_hat.detach().flatten().cpu().numpy())
    print(f"Pearson correlation coefficient: {pr}")

    #计算Spearman相关系数
    corr, _ = spearmanr(mod.y_out.detach().flatten().cpu().numpy(), Y_hat.detach().flatten().cpu().numpy())
    print(f"Spearman correlation coefficient: {corr}")    

    # 保存预测结果
    output_file = ("/home/x_kejli/Y_hat.tsv")
    pd.DataFrame(Y_hat.detach().cpu().numpy()).to_csv(output_file, sep="\t", index=False, header=False)
    print(f"Output saved to {output_file}")


    
  

# 
dict_to_bench={"name": "model_to_test_on_CPU_and_GPU","device":"cuda"}
                          
benchmark_this_dict(dict_to_bench)

