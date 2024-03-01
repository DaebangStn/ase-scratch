# This file aims to process the latent vector in the form of HDF5 file into a kernel density estimation (KDE) model.
import h5py
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler


def apply_pca(data, n_components=2):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)  # Standardize the data
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data_scaled)


def apply_tsne(data, n_components=2, perplexity=30):
    from sklearn.manifold import TSNE
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)  # Standardize the data
    tsne = TSNE(init='random', n_components=n_components, perplexity=perplexity, learning_rate='auto')
    return tsne.fit_transform(data_scaled)


def apply_umap(data, n_components=2):
    import umap
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)  # Standardize the data
    reducer = umap.UMAP(n_components=n_components)
    return reducer.fit_transform(data_scaled)


def apply_lda(data, labels, n_components=2):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

    if labels is None:
        raise ValueError("LDA requires class labels.")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)  # Standardize the data
    reducer = LDA(n_components=n_components)
    return reducer.fit_transform(data_scaled, labels)


def apply_mds(data, n_components=2):
    from sklearn.manifold import MDS
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)  # Standardize the data
    reducer = MDS(n_components=n_components)
    return reducer.fit_transform(data_scaled)


#  Naive visualization of the latent space
def visualize_naive(data, method_name, labels=None, plot_altogether=False):
    if labels is not None:
        plt.scatter(data[:, 0], data[:, 1], cmap='viridis', s=50, alpha=0.6, label=labels)
    else:
        plt.scatter(data[:, 0], data[:, 1], s=50, alpha=0.6)
    if plot_altogether:
        plt.title(f'Latent Space (Naive-{method_name})')
    else:
        plt.title(f'Latent Space for [{labels}] (Naive-{method_name})')

    plt.xlabel('Component 1')
    plt.ylabel('Component 2')


#  Kernel Density Estimation (KDE) visualization of the latent space
def visualize_kde(data_2d, method_name, labels=None, vmin=0.0, vmax=0.005):
    """
    Visualizes the 2D data using Kernel Density Estimation.

    Parameters:
    - data_2d: numpy array of shape (n_samples, 2) after dimensionality reduction
    """
    # Apply KDE
    kde = KernelDensity(bandwidth=0.5, kernel='gaussian')
    kde.fit(data_2d)

    # Create a grid of points for visualization
    x_min, x_max = data_2d[:, 0].min() - 1, data_2d[:, 0].max() + 1
    y_min, y_max = data_2d[:, 1].min() - 1, data_2d[:, 1].max() + 1
    x, y = np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)
    grid_samples = np.array([X.ravel(), Y.ravel()]).T
    log_dens = kde.score_samples(grid_samples)
    Z = np.exp(log_dens).reshape(X.shape)

    # Plot KDE
    # plt.contourf(X, Y, Z, cmap='viridis', levels=50, vmin=vmin, vmax=vmax)
    plt.contourf(X, Y, Z, cmap='viridis', levels=50)
    plt.colorbar(label='Density')
    plt.title(f'Latent Space for [{labels}] (KDE-{method_name})')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')


def build_args():
    parameters = [
        {"name": "--hdf5", "type": str, "default": "",
         "help": "Path to the HDF5 file containing the latent vectors."},
        {"name": "--reduce-alg", "type": str, "default": "tSNE",
         "choices": ["tSNE", "PCA", "UMAP", "LDA", "MDS"],
         "help": "Algorithm to reduce the dimensionality of the latent vectors."},
        {"name": "--reduce-altogether", "type": bool, "default": False,
         "help": "Reduce the dimensionality of the latent without regarding its label."},
        {"name": "--plot-altogether", "type": bool, "default": False,
         "help": "Plot on the single image."},
        {"name": "--visualization", "type": str, "default": "kde", "choices": ["naive", "kde"],
         "help": "Visualization method to use."}
    ]
    parser = argparse.ArgumentParser(description="Visualize Latent Space")
    for param in parameters:
        kwargs = {k: v for k, v in param.items() if k != "name"}
        parser.add_argument(param["name"], **kwargs)

    args = parser.parse_args()
    return args


def main():
    args = build_args()

    if not args.hdf5:
        raise ValueError("Please provide the path to the HDF5 file containing the latent vectors.")

    if args.reduce_alg == "tSNE":
        apply_reduction = apply_tsne
    elif args.reduce_alg == "PCA":
        apply_reduction = apply_pca
    elif args.reduce_alg == "UMAP":
        apply_reduction = apply_umap
    elif args.reduce_alg == "LDA":
        apply_reduction = apply_lda
    elif args.reduce_alg == "MDS":
        apply_reduction = apply_mds
    else:
        raise ValueError(f"Given algorithm({args.reduce_alg}) to reduce the dimensionality of the "
                         f"latent vectors is not supported.")

    if args.visualization == "naive":
        visualize = visualize_naive
    elif args.visualization == "kde":
        visualize = visualize_kde
    else:
        raise ValueError(f"Given visualization method({args.visualization}) is not supported.")

    with h5py.File(args.hdf5, 'r') as f:
        total = len(f.keys())

        if args.reduce_altogether:
            all_data = []
            labels = []
            for key, data in f.items():
                all_data.append(np.array(data))
                labels.extend([key] * len(data))

            concatenated_data = np.concatenate(all_data, axis=0)
            print(f"Reducing {concatenated_data.shape} samples...")
            reduced_data = apply_reduction(concatenated_data)

            reduced_data_dict = {}
            dict_idx = 0

            for key, data in f.items():
                end = dict_idx + len(data)
                reduced_data_dict[key] = reduced_data[dict_idx:end]
                dict_idx = end

            for idx, (key, data) in enumerate(reduced_data_dict.items()):
                if not args.plot_altogether:
                    plt.figure(idx, figsize=(8, 6))
                visualize(data, method_name=args.reduce_alg, labels=key, plot_altogether=args.plot_altogether)
                print(f"Visualizing {key} ({idx + 1}/{total})")

        else:
            for idx, (key, data) in enumerate(f.items()):
                data = np.array(data)
                print(f"Reducing {data.shape} samples...")
                reduced_data = apply_reduction(data)
                plt.figure(idx, figsize=(8, 6))
                visualize(reduced_data, method_name=args.reduce_alg, labels=key)
                print(f"Visualizing {key} ({idx + 1}/{total})")
    if args.plot_altogether:
        plt.legend()
    plt.show(block=False)
    input("Press any key to continue...")
    plt.close()


if __name__ == "__main__":
    main()
