import torch as t
import torch.nn.functional as F


def keras_style_init(tensor, initializer='glorot_uniform'):
    """
    Apply Keras-style initializers to a tensor

    Args:
        tensor: PyTorch tensor to initialize
        initializer: String indicating which initializer to use
            Options: 'glorot_uniform', 'glorot_normal', 'he_normal',
                     'he_uniform', 'zeros', 'ones', 'random_normal', 'random_uniform'

    Returns:
        Initialized tensor
    """
    import math
    import torch.nn.init as init

    # Special handling for bias terms (1D tensors)
    if len(tensor.shape) < 2:
        if initializer == 'zeros':
            init.zeros_(tensor)
        elif initializer == 'ones':
            init.ones_(tensor)
        elif initializer in ['random_normal', 'glorot_normal', 'he_normal']:
            init.normal_(tensor, mean=0, std=0.05)
        elif initializer in ['random_uniform', 'glorot_uniform', 'he_uniform']:
            init.uniform_(tensor, -0.05, 0.05)
        return tensor

    # For weight matrices (2D+ tensors)
    fan_in, fan_out = init._calculate_fan_in_and_fan_out(tensor)

    if initializer == 'glorot_uniform':  # Xavier uniform
        limit = math.sqrt(6 / (fan_in + fan_out))
        init.uniform_(tensor, -limit, limit)
    elif initializer == 'glorot_normal':  # Xavier normal
        std = math.sqrt(2 / (fan_in + fan_out))
        init.normal_(tensor, mean=0, std=std)
    elif initializer == 'he_normal':  # Kaiming normal
        init.kaiming_normal_(tensor, mode='fan_in', nonlinearity='relu')
    elif initializer == 'he_uniform':  # Kaiming uniform
        init.kaiming_uniform_(tensor, mode='fan_in', nonlinearity='relu')
    elif initializer == 'zeros':
        init.zeros_(tensor)
    elif initializer == 'ones':
        init.ones_(tensor)
    elif initializer == 'random_normal':
        init.normal_(tensor, mean=0, std=0.05)
    elif initializer == 'random_uniform':
        init.uniform_(tensor, -0.05, 0.05)
    else:
        raise ValueError(f"Unknown initializer: {initializer}")

    return tensor


def calcRegLoss(model):
    """
    Calculate the regularization loss by summing the L2 norm of model parameters.

    Parameters:
    model (torch.nn.Module): The neural network model.

    Returns:
    ret (float): The regularization loss.
    """
    ret = 0
    for W in model.parameters():
        ret += W.norm(2).square()
    return ret


def contrastLoss(embeds1, embeds2, nodes, temp):
    """
    Calculate the contrastive loss for embeddings.

    Parameters:
    embeds1 (torch.Tensor): The first set of embeddings.
    embeds2 (torch.Tensor): The second set of embeddings.
    nodes (torch.Tensor): Indices of nodes used for contrastive loss.
    temp (float): Temperature parameter for the contrastive loss.

    Returns:
    loss (torch.Tensor): The contrastive loss.
    """
    embeds1 = F.normalize(embeds1 + 1e-8, p=2)
    embeds2 = F.normalize(embeds2 + 1e-8, p=2)
    pckEmbeds1 = embeds1[nodes]
    pckEmbeds2 = embeds2[nodes]
    nume = t.exp(t.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / temp)
    deno = t.exp(pckEmbeds1 @ embeds2.T / temp).sum(-1) + 1e-8
    return -t.log(nume / deno).mean()


def contrastLoss2(embeds1, embeds2, nodes, temp, mode='sym', lambd=5e-3):
    """
    Alternative contrastive losses used by ablation experiments.
    """
    z1 = F.normalize(embeds1 + 1e-8, dim=1)
    z2 = F.normalize(embeds2 + 1e-8, dim=1)

    if mode == 'sym':
        a, b = z1[nodes], z2[nodes]
        logits_ab = (a @ z2.T) / temp
        logits_ba = (b @ z1.T) / temp
        labels = t.arange(a.size(0), device=a.device)
        loss = F.cross_entropy(logits_ab, labels)
        loss += F.cross_entropy(logits_ba, labels)
        return 0.5 * loss

    elif mode == 'barlow':
        a, b = z1[nodes], z2[nodes]
        N, d = a.size()
        c = (a.T @ b) / N
        on_diag = t.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = (c.flatten()[1:].view(d-1, d+1)[:, :-1]).pow_(2).sum()
        return on_diag + lambd * off_diag

    elif mode == 'multi':
        a = z1[nodes]
        b = z2[nodes]

        sim = (a @ b.T) / temp

        ids = nodes.to(sim.device)
        mask = ids.unsqueeze(1).eq(ids.unsqueeze(0))

        sim = sim - sim.max(dim=1, keepdim=True).values
        exp = t.exp(sim)

        pos = (exp * mask).sum(1)
        neg = (exp * (~mask)).sum(1) + 1e-8
        return (-t.log(pos / (pos + neg))).mean()

    else:
        raise ValueError(f"Unknown contrastive mode: {mode}")


def ce(pred, target):
    """
    Calculate the cross-entropy loss between predicted and target values.

    Parameters:
    pred (torch.Tensor): Predicted values.
    target (torch.Tensor): Target values.

    Returns:
    loss (torch.Tensor): The cross-entropy loss.
    """
    return F.cross_entropy(pred, target)


def l2_norm(x):
    """
    Calculate L2 normalization of a tensor.

    Parameters:
    x (torch.Tensor): The input tensor.

    Returns:
    normalized_x (torch.Tensor): The L2 normalized tensor.
    """

    epsilon = t.FloatTensor([1e-12]).to(x.device)

    return x / (t.max(t.norm(x, dim=1, keepdim=True), epsilon))
