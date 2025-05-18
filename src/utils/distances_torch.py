import torch
from tqdm import tqdm


def compute_centroids(train_features, train_labels, class_cond=True):
    """
    Compute centroids in PyTorch

    Args:
        train_features: torch.Tensor (n_samples, n_features)
        train_labels: torch.Tensor (n_samples,)
        class_cond: bool (whether to compute per-class centroids)

    Returns:
        torch.Tensor (n_classes, n_features) if class_cond else (n_features,)
    """
    if class_cond:
        unique_labels = torch.unique(train_labels)
        centroids = []
        for label in unique_labels:
            mask = (train_labels == label)
            centroids.append(train_features[mask].mean(dim=0))
        return torch.stack(centroids)
    else:
        return train_features.mean(dim=0)


def compute_covariance(centroids, train_features, train_labels, class_cond=True):
    """
    Compute inverse covariance matrix in PyTorch

    Args:
        centroids: torch.Tensor (n_classes, n_features) or (n_features,)
        train_features: torch.Tensor (n_samples, n_features)
        train_labels: torch.Tensor (n_samples,)
        class_cond: bool (whether to compute per-class covariance)

    Returns:
        torch.Tensor (n_features, n_features)
    """
    n_features = train_features.size(1)
    device = train_features.device
    cov = torch.zeros((n_features, n_features), device=device, dtype=torch.float64)

    if class_cond:
        for c, mu_c in enumerate(centroids):
            mask = (train_labels == c)
            X_c = train_features[mask] - mu_c.unsqueeze(0)
            cov += X_c.T @ X_c
    else:
        X_centered = train_features - centroids.unsqueeze(0)
        cov = X_centered.T @ X_centered

    cov /= train_features.size(0)

    try:
        sigma_inv = torch.linalg.inv(cov)
    except:
        sigma_inv = torch.linalg.pinv(cov)
        print("Compute pseudo-inverse matrix")

    return sigma_inv


def mahalanobis_distance_marginal(
        train_features, train_labels, eval_features, centroids=None, covariance=None
):
    """
    Compute marginal Mahalanobis distance in PyTorch

    Args:
        train_features: torch.Tensor (n_train, n_features)
        train_labels: torch.Tensor (n_train,)
        eval_features: torch.Tensor (n_eval, n_features)
        centroids: Optional[torch.Tensor] (n_features,)
        covariance: Optional[torch.Tensor] (n_features, n_features)

    Returns:
        torch.Tensor (n_eval,)
    """
    if centroids is None:
        centroids = compute_centroids(train_features, train_labels, class_cond=False)
    if covariance is None:
        covariance = compute_covariance(
            centroids, train_features, train_labels, class_cond=False
        )

    diff = eval_features - centroids.unsqueeze(0)
    dists = torch.einsum('ni,ij,nj->n', diff, covariance, diff)
    return dists


def mahalanobis_distance_relative(
        train_features,
        train_labels,
        eval_features,
        centroids=None,
        covariance=None,
        train_centroid=None,
        train_covariance=None,
):
    """
    Compute relative Mahalanobis distance in PyTorch

    Args:
        train_features: torch.Tensor (n_train, n_features)
        train_labels: torch.Tensor (n_train,)
        eval_features: torch.Tensor (n_eval, n_features)
        centroids: Optional[torch.Tensor] (n_classes, n_features)
        covariance: Optional[torch.Tensor] (n_features, n_features)
        train_centroid: Optional[torch.Tensor] (n_features,)
        train_covariance: Optional[torch.Tensor] (n_features, n_features)

    Returns:
        torch.Tensor (n_eval,)
    """
    if centroids is None:
        centroids = compute_centroids(train_features, train_labels)
    if covariance is None:
        covariance = compute_covariance(centroids, train_features, train_labels)

    # Compute class-conditional distances
    diff = eval_features.unsqueeze(1) - centroids.unsqueeze(0)  # (n_eval, n_classes, n_features)
    dists = torch.einsum('nci,ij,ncj->nc', diff, covariance, diff)  # (n_eval, n_classes)

    # Compute marginal distances
    if train_centroid is None:
        train_centroid = compute_centroids(train_features, train_labels, class_cond=False)
    if train_covariance is None:
        train_covariance = compute_covariance(train_centroid, train_features, train_labels, class_cond=False)

    diff_marginal = eval_features - train_centroid.unsqueeze(0)  # (n_eval, n_features)
    md_marginal = torch.einsum('ni,ij,nj->n', diff_marginal, train_covariance, diff_marginal)  # (n_eval,)

    # Compute relative distances and return minimum
    relative_dists = dists - md_marginal.unsqueeze(1)  # (n_eval, n_classes)
    return torch.min(relative_dists, dim=1)[0]  # (n_eval,)


def mahalanobis_distance(
        train_features,
        train_labels,
        eval_features,
        centroids=None,
        covariance=None,
        return_full=False,
):
    """
    Compute Mahalanobis distance in PyTorch

    Args:
        train_features: torch.Tensor (n_train, n_features)
        train_labels: torch.Tensor (n_train,)
        eval_features: torch.Tensor (n_eval, n_features)
        centroids: Optional[torch.Tensor] (n_classes, n_features)
        covariance: Optional[torch.Tensor] (n_features, n_features)
        return_full: bool (whether to return full distance matrix)

    Returns:
        torch.Tensor (n_eval,) or (n_eval, n_classes)
    """
    if centroids is None:
        centroids = compute_centroids(train_features, train_labels)
    if covariance is None:
        covariance = compute_covariance(centroids, train_features, train_labels)

    diff = eval_features.unsqueeze(1) - centroids.unsqueeze(0)
    dists = torch.einsum('nci,ij,ncj->nc', diff, covariance, diff)

    if return_full:
        return dists
    else:
        return torch.min(dists, dim=1)[0]