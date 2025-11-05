import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
import torch

# ──────────────────────────────────────────────────────────────
# Ensure the LEMBAS package is importable (project_root in PYTHONPATH)
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from LEMBAS.model.bionetwork import format_network, SignalingModel
from LEMBAS.model.train import train_signaling_model
import LEMBAS.utilities as utils

# ──────────────────────────────────────────────────────────────
# Core benchmarking function
# ──────────────────────────────────────────────────────────────

def benchmark_this_dict(dict_to_bench: dict):
    """Train a SignalingModel and report Pearson/MSE/R² on the test set.

    Parameters
    ----------
    dict_to_bench : dict
        Mini‑config holding (at minimum) a "device" key ("cpu" | "cuda").
    """

    # 1. Resources & reproducibility
    n_cores = 12
    utils.set_cores(n_cores)

    seed = 888
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    utils.set_seeds(seed=seed)

    device = dict_to_bench.get("device", "cpu")
    print(f"Current device: {device}")

    # 2. Data loading
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
    net          = pd.read_csv(os.path.join(data_path, "network.tsv"),        sep="\t", index_col=False)
    ligand_input = pd.read_csv(os.path.join(data_path, "TF.tsv"),             sep="\t", low_memory=False)
    tf_output    = pd.read_csv(os.path.join(data_path, "Geneexpression.tsv"), sep="\t", low_memory=False)

    net = format_network(net, weight_label="Interaction")

    # 3. Hyper‑parameters
    projection_amplitude_in  = 1.0
    projection_amplitude_out = 1.2

    bionet_params = {
        "target_steps": 150,
        "max_steps": 10,
        "exp_factor": 50,
        "tolerance": 1e-20,
        "leak": 1e-2,
    }

    lr_params = {
        "max_iter": 600,
        "learning_rate": 2e-4,
    }

    other_params = {
        "batch_size": 25,
        "noise_level": 10,
        "gradient_noise_level": 1e-9,
    }

    regularization_params = {
        "param_lambda_L2": 1e-6,
        "moa_lambda_L1": 0.1,
        "ligand_lambda_L2": 1e-5,
        "uniform_lambda_L2": 1e-4,
        "uniform_max": 1 / projection_amplitude_out,
        "spectral_loss_factor": 1e-5,
    }

    spectral_radius_params = {
        "n_probes_spectral": 5,
        "power_steps_spectral": 10,
        "subset_n_spectral": 10,
    }

    hyper_params = {
        **lr_params,
        **other_params,
        **regularization_params,
        **spectral_radius_params,
    }

    # 4. Build model
    mod = SignalingModel(
        net=net,
        X_in=ligand_input,
        y_out=tf_output,
        projection_amplitude_in=projection_amplitude_in,
        projection_amplitude_out=projection_amplitude_out,
        weight_label="Interaction",
        source_label="TF",
        target_label="Gene",
        bionet_params=bionet_params,
        dtype=torch.float32,
        device=device,
        seed=seed,
    )

    # Convert inputs to tensors (in‑place for mod)
    mod.X_in = mod.df_to_tensor(mod.X_in)
    mod.y_out = mod.df_to_tensor(mod.y_out)

    # Freeze / un‑freeze parameters as desired
    mod.input_layer.weights.requires_grad = True

    loss_fn = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam

    print("\n──────── Training starts ────────")
    mod, cur_loss, cur_eig, mean_loss, stats, \
        X_train, X_test, X_val, y_train, y_test, y_val, train_dataloader = train_signaling_model(
            mod,
            optimizer,
            loss_fn,
            reset_epoch=200,
            hyper_params=hyper_params,
            train_seed=seed,
            verbose=True,
            dict_to_bench=dict_to_bench,
        )

    # 5. Save checkpoints
    torch.save({"model_state_dict": mod.state_dict()}, "full_model_checkpoint.pth")
    torch.save(
        {
            "gatbio_state_dict": mod.signaling_network.state_dict(),
            "custom_parameters": {"edge_weight": mod.signaling_network.edge_weight},
        },
        "signaling_network_full.pth",
    )

    # 新增：保存整个模型为.pt文件
    torch.save(mod, "gat_signaling_model.pt")



# ──────────────────────────────────────────────────────────────
# Convenience CLI entry‑point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    example_cfg = {"name": "model_to_test_on_CPU_and_GPU", "device": "cuda" if torch.cuda.is_available() else "cpu"}
    benchmark_this_dict(example_cfg)



