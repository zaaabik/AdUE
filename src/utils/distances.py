import torch
import numpy as np
from sklearn.decomposition import KernelPCA
from tqdm import tqdm
from sklearn.covariance import MinCovDet

DOUBLE_INFO = torch.finfo(torch.double)
JITTERS = [10**exp for exp in range(-15, 0, 1)]


def mahalanobis_distance_with_known_centroids_sigma_inv(
    centroids, centroids_mask, sigma_inv, eval_features
):
    """
    - This function takes in centroids, centroids_mask, sigma_inv, and eval_features.
    - tensor of Mahalanobis distances is returned.
    """
    # step 1: calculate the difference (diff) between each evaluation feature and each centroid by subtracting the centoids from the features.

    diff = eval_features.unsqueeze(1) - centroids.unsqueeze(
        0
    )  # bs (b), num_labels (c / s), dim (d / a)

    # step 2: the Mahalanobis distance is computed using the formula: sqrt(diff @ sigmainv @ diff),
    #  where diff is reshaped to match the dimensions of sigmainv.

    dists = torch.sqrt(torch.einsum("bcd,da,bsa->bcs", diff, sigma_inv, diff))
    device = dists.device

    # step 3: obtain a tensor of distances for each evaluation feature and centroid pair.

    dists = torch.stack([torch.diag(dist).cpu() for dist in dists], dim=0)

    # If centroids_mask is not None, the distances corresponding to masked centroids are filled with infinity.

    if centroids_mask is not None:
        dists = dists.masked_fill_(centroids_mask, float("inf")).to(device)
    return dists  # np.min(dists, axis=1)


def mahalanobis_distance_marginal(
    train_features, train_labels, eval_features, centroids=None, covariance=None
):
    if centroids is None:
        centroids = compute_centroids(train_features, train_labels, class_cond=False)
    if covariance is None:
        covariance = compute_covariance(
            centroids, train_features, train_labels, class_cond=False
        )

    diff = eval_features - centroids[None, :]
    dists = np.matmul(np.matmul(diff, covariance), diff.T)
    return np.diag(dists)


def compute_centroids(train_features, train_labels, class_cond=True):
    if class_cond:
        centroids = []
        for label in np.sort(np.unique(train_labels)):
            centroids.append(train_features[train_labels == label].mean(axis=0))
        return np.asarray(centroids)
    else:
        return train_features.mean(axis=0)


def compute_covariance(centroids, train_features, train_labels, class_cond=True):
    cov = np.zeros((train_features.shape[1], train_features.shape[1]))
    if class_cond:
        for c, mu_c in tqdm(enumerate(centroids)):
            for x in train_features[train_labels == c]:
                d = (x - mu_c)[:, None]
                cov += d @ d.T
    else:
        for x in train_features:
            d = (x - centroids)[:, None]
            cov += d @ d.T
    cov /= train_features.shape[0]

    try:
        sigma_inv = np.linalg.inv(cov)
    except:
        sigma_inv = np.linalg.pinv(cov)
        print("Compute pseudo-inverse matrix")

    return sigma_inv


def mahalanobis_distance_relative(
    train_features,
    train_labels,
    eval_features,
    centroids=None,
    covariance=None,
    train_centroid=None,
    train_covariance=None,
):
    if centroids is None:
        centroids = compute_centroids(train_features, train_labels)
    if covariance is None:
        covariance = compute_covariance(centroids, train_features, train_labels)

    diff = eval_features[:, None, :] - centroids[None, :, :]
    dists = np.matmul(np.matmul(diff, covariance), diff.transpose(0, 2, 1))
    dists = np.asarray([np.diag(dist) for dist in dists])

    md_marginal = mahalanobis_distance_marginal(
        train_features, train_labels, eval_features, train_centroid, train_covariance
    )
    return np.min(dists - md_marginal[:, None], axis=1)


def mahalanobis_distance(
    train_features,
    train_labels,
    eval_features,
    centroids=None,
    covariance=None,
    return_full=False,
):
    if centroids is None:
        centroids = compute_centroids(train_features, train_labels)
    if covariance is None:
        covariance = compute_covariance(centroids, train_features, train_labels)

    diff = eval_features[:, None, :] - centroids[None, :, :]

    dists = np.matmul(np.matmul(diff, covariance), diff.transpose(0, 2, 1))
    dists = np.asarray([np.diag(dist) for dist in dists])

    if return_full:
        return dists
    else:
        return np.min(dists, axis=1)


def MCD_covariance(X, y=None, label=None, seed=42, support_fraction=None):
    try:
        if label is None:
            cov = MinCovDet(random_state=seed, support_fraction=support_fraction).fit(X)
        else:
            cov = MinCovDet(random_state=seed, support_fraction=support_fraction).fit(X[y == label])
    except ValueError:
        print(
            "****************Try fitting covariance with support_fraction=0.9 **************"
        )
        try:
            if label is None:
                cov = MinCovDet(random_state=seed, support_fraction=0.9).fit(X)
            else:
                cov = MinCovDet(random_state=seed, support_fraction=0.9).fit(
                    X[y == label]
                )
        except ValueError:
            print(
                "****************Try fitting covariance with support_fraction=1.0 **************"
            )
            if label is None:
                cov = MinCovDet(random_state=seed, support_fraction=1.0).fit(X)
            else:
                cov = MinCovDet(random_state=seed, support_fraction=1.0).fit(
                    X[y == label]
                )
    return cov


def rde_distance(
        train_features,
        test_features,
        n_components=256,
        support_fraction=None

):
    pca = KernelPCA(n_components=n_components, kernel="rbf", random_state=42)
    train_features_pca = pca.fit_transform(train_features.float().cpu().numpy())
    test_features_pca = pca.transform(test_features.float().cpu().numpy())

    mcd = MCD_covariance(train_features_pca, seed=42, support_fraction=support_fraction)
    rde_distance = mcd.mahalanobis(test_features_pca)
    return rde_distance