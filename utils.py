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

# directory = 'output'

# # for name in os.listdir(directory):
# #     if 
# #     clse = np.load(directory + "/" + name)
# #     import pdb; pdb.set_trace()
# #     print(clse.shape)

# input_dir = directory
# token_pooling = "avg"
# ood_datasets = "20news,trec,wmt16"
# #ood_datasets = "trec"

# ind_train_features = np.load(
#     '{}/{}_ind_train_features.npy'.format(input_dir, token_pooling))
# num_layers = ind_train_features.shape[0] - 1
# ind_train_labels = np.load('{}/{}_ind_train_labels.npy'.format(input_dir, token_pooling))
# ind_test_features = np.load('{}/{}_ind_test_features.npy'.format(input_dir, token_pooling))
# ind_test_labels = np.load('{}/{}_ind_test_labels.npy'.format(input_dir, token_pooling))
# ind_test_scores_list = []
# ood_test_scores_list = [[] for _ in range(len(ood_datasets.split(',')))]
# ood_test_features_list = []
# for ood_dataset in ood_datasets.split(','):
#     ood_test_features = np.load(
#         '{}/{}_ood_features_{}.npy'.format(input_dir, token_pooling, ood_dataset))
#     ood_test_features_list.append(ood_test_features)


def plot_PCA(ind_test_features, ind_train_features, ood_features_list, mode = 'avg'):
    pca = PCA(n_components=2)
    if mode == 'avg':
        ind_train = np.mean(ind_train_features, axis = 0)
        ind_test = np.mean(ind_test_features, axis = 0)
        ood_list = [np.mean(ood_test, axis = 0) for ood_test in ood_test_features_list]
    else:
        ind_train = ind_train_features[-1]
        ind_test = ind_test_features[-1]
        ood_list = [ood_test[-1] for ood_test in ood_test_features_list]

    X_pca = pca.fit_transform(ind_train)

    # Plot the results
    
    X_pca = pca.transform(ind_test)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c = 'r', alpha = 0.5)
    for ood_test in ood_list:
        X_pca = pca.transform(ood_test)
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha = 0.5)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()

def new_plot_PCA(ind_test_features, ind_test_labels, ind_train_features, ood_features, save = False, i = str(-1)):
    pca = PCA(n_components=2)
    mean = np.mean(ind_train_features, axis = 0)
    var = np.var(ind_train_features, axis = 0)
    scaled_data = (ind_train_features - mean)/var
    pca.fit(scaled_data)
    scaled_data = (ind_test_features - mean)/var
    embedding = pca.transform(scaled_data)
    fig, ax = plt.subplots()
    ax.scatter(
    embedding[:, 0],
    embedding[:, 1], c = ind_test_labels, alpha = 0.2)
    scaled_data = (ood_features - mean)/var
    embedding = pca.transform(scaled_data)
    ax.scatter(
    embedding[:, 0],
    embedding[:, 1], c = 'blue', alpha = 0.2)
    if save:
        fig.savefig('PCA_mean' + str(i) + '.png', dpi=300)
        return
    ax.show()

def plot_UMAP(ind_test_features, ind_test_labels, ind_train_features, ood_features, save = False, i = str(-1)):
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
    scaled_data = (ood_features - mean)/var
    embedding = reducer.transform(scaled_data)
    ax.scatter(
    embedding[:, 0],
    embedding[:, 1], c = 'blue', alpha = 0.2)
    if save:
        fig.savefig('Umap_mean' + str(i) + '.png', dpi=300)
        return
    ax.show()

#plot_PCA(ind_test_features, ind_train_features, ood_test_features_list, mode = 'avg')

def get_distance_score(class_mean, precision, features, measure='maha'):
    num_classes = len(class_mean)
    num_samples = len(features)
    class_mean = [torch.from_numpy(m).float() for m in class_mean]
    precision = torch.from_numpy(precision).float()
    features = torch.from_numpy(features).float()
    scores = []
    for c in range(num_classes):
        centered_features = features.data - class_mean[c]
        if measure == 'maha':
            score = -1.0 * torch.mm(torch.mm(centered_features, precision),
                    centered_features.t()).diag()
        elif measure == 'euclid':
            score = -1.0*torch.mm(centered_features,
                centered_features.t()).diag()
        elif measure == 'cosine':
            score = torch.tensor([CosineSimilarity()(features[i].reshape(
                1, -1), class_mean[c].reshape(1, -1)) for i in range(num_samples)])
        else:
            raise ValueError("Unknown distance measure")
        scores.append(score.reshape(-1, 1))
    scores = torch.cat(scores, dim=1)  # num_samples, num_classes
    scores, _ = torch.max(scores, dim=1)  # num_samples
    scores = scores.cpu().numpy()
    return scores

def get_distance_irw(features, train, n_proj=100):
    features = torch.from_numpy(features).float()
    train = torch.from_numpy(train).float()
    x = torch.randn(n_proj, features.shape[1])
    norms = torch.norm(x, dim=1, keepdim=True)
    v = x / norms
    tot = torch.zeros(features.shape[0])
    test = torch.zeros(features.shape[0])
    for j, v_j in enumerate(tqdm(v)): 
        inner_product = v_j@features.T
        inf = (inner_product[:, None] <= torch.mm(train, v_j[:, None]).T).float().mean(dim=1)
        tot += torch.min(inf, 1 - inf)
    scores = tot / n_proj
    scores = scores.cpu().numpy()
    return scores
     


def sample_estimator(features, labels):
    labels = labels.reshape(-1)
    num_classes = np.unique(labels).shape[0]
    #group_lasso = EmpiricalCovariance(assume_centered=False)
    #group_lasso =  MinCovDet(assume_centered=False, random_state=42, support_fraction=1.0)
    # ShurunkCovariance is more stable and robust where the condition number is large
    group_lasso = sklearn.covariance.ShrunkCovariance()
    sample_class_mean = []
    for c in range(num_classes):
        current_class_mean = np.mean(features[labels == c, :], axis=0)
        sample_class_mean.append(current_class_mean)
    X = [features[labels == c, :] - sample_class_mean[c] for c in range(num_classes)]
    X = np.concatenate(X, axis=0)
    group_lasso.fit(X)
    precision = group_lasso.precision_
    mean_tot = [np.mean(features, axis = 0)]
    return sample_class_mean, precision, mean_tot

def all_UMAP():
    for i in range(len(ind_test_features)):
        plot_UMAP(ind_test_features[i], ind_test_labels, ind_train_features[i], ood_test_features_list[2][i], True, i)


def test_UMAP():
    for i in range(1, len(ind_test_features)):
        plot_UMAP(np.mean(ind_test_features[:i], axis = 0), ind_test_labels, np.mean(ind_train_features[:i], axis = 0), np.mean(ood_test_features_list[0][:i], axis = 0), True, i)
   


# new_plot_PCA(np.mean(ind_test_features, axis = 0), ind_test_labels, np.mean(ind_train_features, axis = 0), np.mean(ood_test_features_list[2], axis = 0), True, 0)
# ood_test_scores_list = [[] for _ in range(len(ood_datasets.split(',')))]
# ood_irw_score_list = [[] for _ in range(len(ood_datasets.split(',')))]
# ind_test_score = []
# num_layers = ind_train_features.shape[0] - 1
# maha = True
# avg = True

# if maha:
#     for layer in tqdm(range(0, num_layers+1)):
#         ind_test_score.append(get_distance_irw(ind_test_features[layer], ind_train_features[layer]))
        
#         # sample_class_mean, precision, mean_tot = sample_estimator(ind_train_features[layer], ind_train_labels)
#         # ind_test_score.append(get_distance_score(sample_class_mean, precision, ind_test_features[layer]) + get_distance_score(mean_tot, precision, ind_test_features[layer]) )
#         for i, ood_test in enumerate(ood_test_features_list):
#             ood_test_scores_list[i].append(get_distance_irw(ood_test[layer], ind_train_features[layer]))
#             #ood_test_scores_list[i].append(get_distance_score(sample_class_mean, precision, ood_test[layer]) + get_distance_score(mean_tot, precision, ood_test[layer]))
#     # metrics_tot = []
#     # for i in range(len(ind_test_features)):
#     #     metrics_tot.append(get_metrics(ind_test_score[i], ood_test_scores_list[0][i]))


# if avg:
#     ind_irw_score = get_distance_irw(np.mean(ind_test_features, axis = 0), np.mean(ind_train_features, axis = 0))
#     # sample_class_mean, precision, mean_tot = sample_estimator(np.mean(ind_train_features[:9], axis = 0) , ind_train_labels)
#     # ind_test_score.append(get_distance_score(sample_class_mean, precision, np.mean(ind_test_features[:9], axis = 0)) + get_distance_score(mean_tot, precision, np.mean(ind_test_features[:9], axis = 0) ))
#     for i, ood_test in enumerate(ood_test_features_list):
#         ood_irw_score_list[i].append(get_distance_irw(np.mean(ood_test, axis = 0), np.mean(ind_train_features, axis = 0)))
#         #ood_test_scores_list[i].append(get_distance_score(sample_class_mean, precision, np.mean(ood_test[:9], axis = 0)) + get_distance_score(mean_tot, precision, np.mean(ood_test[:9], axis = 0) ))





def plot_hist(ood, ind):
    xmin = min(np.min(ood), np.min(ind))
    xmax = max(np.max(ood), np.max(ind))
    bin_edges = np.linspace(xmin, xmax, 40)

    # Plot the histograms with the same bin edges
    plt.hist(ood, color='red', alpha=.5, bins=bin_edges)
    plt.hist(ind, color='blue', alpha=.5, bins=bin_edges)

    plt.show()

def metrics_to_plot(ind_test_score, ood_test_scores_list, ood_mean, ind_mean):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
    colors = ['r', 'g', 'b']
    for j in range(len(ood_test_scores_list)):
        metrics_tot = []
        AUROC = []
        FPR = []
        for i in range(len(ind_test_features)):
            metrics= get_metrics(ind_test_score[i], ood_test_scores_list[j][i])
            metrics_tot.append(metrics)
            AUROC.append(metrics['AUROC'])
            FPR.append(metrics['FPR@tpr=0.95'])
        ax1.plot(AUROC, label = ood_datasets.split(',')[j], c = colors[j])
        ax2.plot(FPR, label = ood_datasets.split(',')[j], c = colors[j])
        metrics_mean = get_metrics(ind_mean, ood_mean[j][0])
        ax1.axhline(metrics_mean['AUROC'], linestyle = 'dashed', c = colors[j])
        ax2.axhline(metrics_mean['FPR@tpr=0.95'], linestyle = 'dashed', c = colors[j])

    ax1.legend()
    ax2.legend()
    ax1.set_title("AUROC")
    ax2.set_title("FPR@tpr = 0.95")
    fig.savefig('metrics/metrics_' + str(np.random.random()) + '.png', dpi=300)
    plt.show()

# import pdb; pdb.set_trace() 
# metrics_to_plot(ind_test_score, ood_test_scores_list, ood_irw_score_list, ind_irw_score)


# plt.hist(np.mean(ood_test_scores_list[0], axis = 0), color = 'red', alpha = .5)
# plt.hist(np.mean(ind_test_score, axis = 0), color = 'blue', alpha = .5)
# plt.show()

# plt.hist(ood_test_scores_list[0][1], color = 'red', alpha = .5)
# plt.hist(ind_test_score[1], color = 'blue', alpha = .5)
# plt.show()

# plot_hist(ood_test_scores_list[0], ind_test_score)
#import pdb; pdb.set_trace()


# for layer in range(1, num_layers+1):
#     sample_class_mean, precision = sample_estimator(
#         ind_train_features[layer], ind_train_labels)
#     ind_scores = get_distance_score(sample_class_mean, precision,
#                                     ind_test_features[layer], measure=args.distance_metric)
#     ind_test_scores_list.append(ind_scores)
#     for i, ood_dataset in enumerate(args.ood_datasets.split(',')):
#         ood_scores = get_distance_score(sample_class_mean, precision,
#                                         ood_test_features_list[i][layer], measure=args.distance_metric)
#         ood_test_scores_list[i].append(ood_scores)
# ind_test_scores_list = np.transpose(
#     np.array(ind_test_scores_list), (1, 0))  # num_samples, layers
# ood_test_scores_list = [np.transpose(
#     np.array(scores), (1, 0)) for scores in ood_test_scores_list]
# ind_test_scores = np.sum(ind_test_scores_list, axis=1)
# ood_metrics_list = []
# for i, ood_dataset in enumerate(args.ood_datasets.split(',')):
#     ood_test_scores = np.sum(ood_test_scores_list[i], axis=1)
#     metrics = get_metrics(ind_test_scores, ood_test_scores)
#     logger.info('ood dataset: {}'.format(ood_dataset))
#     logger.info('metrics: {}'.format(metrics))
#     ood_metrics_list.append(metrics)
# mean_metrics = {}
# for k, v in metrics.items():
#     mean_metrics[k] = sum(
#         [m[k] for m in ood_metrics_list])/len(ood_metrics_list)
# logger.info('mean metrics: {}'.format(mean_metrics))
