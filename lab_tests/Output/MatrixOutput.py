import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def save_epoch_matrix(data_dir, epoch, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    matrices = []
    for path in npz_files:
        with np.load(path, allow_pickle=True) as data:
            arr = data["hidden_matrices"]
            matrices.append(arr[epoch])
    avg_matrix = np.mean(np.stack(matrices, axis=0), axis=0)
    epoch_dir = os.path.join(save_dir, f"epoch_{epoch}")
    save_matrix(avg_matrix, epoch_dir)

def save_matrix(matrix, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    n = matrix.shape[0] // 2
    mod = matrix[:n, :n]
    lat = matrix[n:, n:]

    for name, m in [("mod", mod), ("lat", lat)]:
        plt.figure()
        plt.imshow(m, vmin=0, vmax=1, origin="lower")
        plt.colorbar()
        plt.title(name)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{name}.png"), dpi=300)
        plt.close()
