import numpy as np
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.covariance import EmpiricalCovariance, MinCovDet
import sklearn.covariance
import torch
from tqdm import tqdm
import umap
from sklearn.preprocessing import StandardScaler
from lib.metrics import get_metrics

def plot_PCA(ind_test_features, ind_test_labels, ind_train_features, ood_features, save=False, i=str(-1)):
    """Plots the results of PCA applied to the input data.
    
    Args:
        ind_test_features (np.ndarray): In-distribution test features.
        ind_test_labels (np.ndarray): In-distribution test labels.
        ind_train_features (np.ndarray): In-distribution train features.
        ood_features (np.ndarray): Out-of-distribution test features.
        save (bool): Whether to save the plot or not.
        i (str): Optional identifier to add to the saved file.
    """
    pca = PCA(n_components=2)
    scaler = StandardScaler()
    scaler.fit(ind_train_features)
    scaled_data = scaler.transform(ind_train_features)
    pca.fit(scaled_data)
    scaled_data = scaler.transform(ind_test_features)
    embedding = pca.transform(scaled_data)
    fig, ax = plt.subplots()
    ax.scatter(embedding[:, 0], embedding[:, 1], c=ind_test_labels, alpha=0.2)
    scaled_data = scaler.transform(ood_features)
    embedding = pca.transform(scaled_data)
    ax.scatter(embedding[:, 0], embedding[:, 1], c='blue', alpha=0.2)
    if save:
        fig.savefig(f'PCA_mean{i}.png', dpi=300)
        plt.close()  # close the figure to avoid memory leaks
    else:
        plt.show()

def plot_UMAP(ind_test_features, ind_test_labels, ind_train_features, ood_features, save=False, i=str(-1)):
    """Plots the results of UMAP applied to the input data.
    
    Args:
        ind_test_features (np.ndarray): In-distribution test features.
        ind_test_labels (np.ndarray): In-distribution test labels.
        ind_train_features (np.ndarray): In-distribution train features.
        ood_features (np.ndarray): Out-of-distribution test features.
        save (bool): Whether to save the plot or not.
        i (str): Optional identifier to add to the saved file.
    """
    reducer = umap.UMAP()
    scaler = StandardScaler()
    scaler.fit(ind_train_features)
    scaled_data = scaler.transform(ind_train_features)
    reducer.fit(scaled_data)
    scaled_data = scaler.transform(ind_test_features)
    embedding = reducer.transform(scaled_data)
    fig, ax = plt.subplots()
    ax.scatter(embedding[:, 0], embedding[:, 1], c=ind_test_labels, alpha=0.2)
    scaled_data = scaler.transform(ood_features)
    embedding = reducer.transform(scaled_data)
    ax.scatter(embedding[:, 0], embedding[:, 1], c='blue', alpha=0.2)
    if save:
        fig.savefig(f'Umap_mean{i}.png', dpi=300)
        plt.close()  # close the figure to avoid memory leaks
    else:
        plt.show()

def plot_UMAP_fpr(ind_test_score, ood_test_score, ind_test_features, ind_test_labels, ind_train_features, ood_features, save = False, i = str(-1)):
    from lib.metrics import get_is_pos
    mask = get_is_pos(ind_test_score, ood_test_score, "largest2smallest")
    reducer = umap.UMAP()
    mean = np.mean(ind_train_features, axis = 0)
    var = np.var(ind_train_features, axis = 0)
    scaled_data = (ind_train_features - mean)/var
    reducer.fit(scaled_data)
    scaled_data = (ind_test_features - mean)/var
    embedding = reducer.transform(scaled_data)
    
    fig, ax = plt.subplots()
    ax.scatter(
    embedding[:, 0],
    embedding[:, 1], c = ind_test_labels, alpha = 0.2)
    mask_1 = mask[:len(embedding)]
    embedding = embedding[~mask_1]
    ax.scatter(
    embedding[:, 0],
    embedding[:, 1], c = 'r', alpha = 0.8 )

    scaled_data = (ood_features - mean)/var
    embedding = reducer.transform(scaled_data)
    ax.scatter(
    embedding[:, 0],
    embedding[:, 1], c = 'blue', alpha = 0.2)

    mask_2 = mask[len(ind_test_score):]
    embedding = embedding[mask_2]
    ax.scatter(
    embedding[:, 0],
    embedding[:, 1], c = 'g', alpha = 0.8 )
    
    if save:
        fig.savefig(f'Umap_mean{i}.png', dpi=300)
        plt.close()  # close the figure to avoid memory leaks
    else:
        plt.show()

def get_distance_score(class_mean, precision, features, measure='maha'):
    """ Calculate distance scores between features and class means
        class_mean: list of mean vectors for each class
        precision: precision matrix for Mahalanobis distance calculation
        features: array of features to compare to class means
        measure: distance measure to use (maha, euclid, or cosine)
    """
    
    # Convert inputs to PyTorch tensors
    num_classes = len(class_mean)
    num_samples = len(features)
    class_mean = [torch.from_numpy(m).float() for m in class_mean]
    precision = torch.from_numpy(precision).float()
    features = torch.from_numpy(features).float()
    
    # Initialize empty list for scores
    scores = []
    
    # Iterate over classes
    for c in range(num_classes):
        centered_features = features.data - class_mean[c]
        # Calculate distance score based on chosen measure
        if measure == 'maha': # Mahalanobis distance
            score = -1.0 * torch.mm(torch.mm(centered_features, precision),
                    centered_features.t()).diag()
        elif measure == 'euclid': # Euclidean distance
            score = -1.0*torch.mm(centered_features,
                centered_features.t()).diag()
        elif measure == 'cosine': # Cosine similarity
            score = torch.tensor([CosineSimilarity()(features[i].reshape(
                1, -1), class_mean[c].reshape(1, -1)) for i in range(num_samples)])
        else:
            raise ValueError("Unknown distance measure")
        scores.append(score.reshape(-1, 1))
    
    # Concatenate scores across classes
    scores = torch.cat(scores, dim=1)  # num_samples, num_classes
    
    # Take maximum score across classes for each sample
    scores, _ = torch.max(scores, dim=1)  # num_samples
    
    # Convert scores to numpy array and return
    scores = scores.cpu().numpy()
    return scores


def get_distance_irw(features, train, n_proj=100):
    """Calculate distance scores using importance-weighted random projections
       features: array of features to compare to training set
        train: array of training set features to compare to
        n_proj: number of random projections to use
    """
    # Convert inputs to PyTorch tensors
    features = torch.from_numpy(features).float()
    train = torch.from_numpy(train).float()
    
    # Generate random projection vectors and normalize
    x = torch.randn(n_proj, features.shape[1])
    norms = torch.norm(x, dim=1, keepdim=True)
    v = x / norms
    
    # Initialize arrays for distance scores
    tot = torch.zeros(features.shape[0])
    test = torch.zeros(features.shape[0])
    
    # Iterate over random projection vectors
    for j, v_j in enumerate(tqdm(v)): 
        # Calculate inner product between random projection vector and features
        inner_product = v_j@features.T
        
        # Calculate importance weights based on whether each feature is closer to the training set or the test set
        inf = (inner_product[:, None] <= torch.mm(train, v_j[:, None]).T).float().mean(dim=1)
        tot += torch.min(inf, 1 - inf)
    
    # Average distance scores across random projections
    scores = tot / n_proj
    
    # Convert scores to numpy array and return
    scores = scores.cpu().numpy()
    return scores


def sample_estimator(features, labels):
    """
    Given training features and labels, compute the sample class mean and precision matrix using the group lasso estimator.
    Args:
        - features (numpy array): training features
        - labels (numpy array): corresponding training labels
    Returns:
        - sample_class_mean (list): list of sample class means for each class
        - precision (numpy array): precision matrix computed using the group lasso estimator
        - mean_tot (list): list with only one element, which is the mean of all features across all classes
    """
    labels = labels.reshape(-1)
    num_classes = np.unique(labels).shape[0]
    group_lasso = sklearn.covariance.ShrunkCovariance()  # use the group lasso estimator for covariance
    sample_class_mean = []
    for c in range(num_classes):
        current_class_mean = np.mean(features[labels == c, :], axis=0)
        sample_class_mean.append(current_class_mean)
    X = [features[labels == c, :] - sample_class_mean[c] for c in range(num_classes)]
    X = np.concatenate(X, axis=0)
    group_lasso.fit(X)
    precision = group_lasso.precision_  # compute the precision matrix
    mean_tot = [np.mean(features, axis=0)]  # compute the mean across all classes
    return sample_class_mean, precision, mean_tot


def plot_hist(ood, ind):
    """
    Plot two histograms side by side for the given OOD and in-distribution scores.
    Args:
        - ood (numpy array): OOD test scores
        - ind (numpy array): in-distribution test scores
    """
    xmin = min(np.min(ood), np.min(ind))
    xmax = max(np.max(ood), np.max(ind))
    bin_edges = np.linspace(xmin, xmax, 40)  # create equally spaced bins for the histograms

    # Plot the histograms with the same bin edges
    plt.hist(ood, color='red', alpha=.5, bins=bin_edges)
    plt.hist(ind, color='blue', alpha=.5, bins=bin_edges)

    plt.show()


def metrics_to_plot(ind_test_score, ood_test_scores_list, ood_mean, ind_mean, ood_datasets):
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
    # Define a list of colors to use for plotting
    colors = ['r', 'g', 'b']
    # Loop over the out-of-distribution test scores and plot the metrics
    for j in range(len(ood_test_scores_list)):
        # Initialize empty lists for storing the metrics
        metrics_tot = []
        AUROC = []
        FPR = []
        # Loop over the individual test features and compute the metrics
        for i in range(len(ind_test_score)):
            # Get the metrics for the current individual and out-of-distribution test features
            metrics= get_metrics(ind_test_score[i], ood_test_scores_list[j][i])
            # Append the metrics to the list of all metrics
            metrics_tot.append(metrics)
            # Append the AUROC and FPR to their respective lists
            AUROC.append(metrics['AUROC'])
            FPR.append(metrics['FPR@tpr=0.95'])
        # Plot the AUROC and FPR for the current out-of-distribution test scores
        ax1.plot(AUROC, label = ood_datasets.split(',')[j], c = colors[j])
        ax2.plot(FPR, label = ood_datasets.split(',')[j], c = colors[j])
        # Get the metrics for the mean of the individual and out-of-distribution test features
        metrics_mean = get_metrics(ind_mean, ood_mean[j][0])
        # Add a dashed horizontal line at the mean AUROC and FPR for the current out-of-distribution test scores
        ax1.axhline(metrics_mean['AUROC'], linestyle = 'dashed', c = colors[j])
        ax2.axhline(metrics_mean['FPR@tpr=0.95'], linestyle = 'dashed', c = colors[j])
    # Add legends to the subplots
    ax1.legend()
    ax2.legend()
    # Set the titles of the subplots
    ax1.set_title("AUROC")
    ax2.set_title("FPR@tpr = 0.95")
    # Save the figure to a file with a random filename and display it
    fig.savefig('metrics_' + str(np.random.random()) + '.png', dpi=300)
    plt.show()
