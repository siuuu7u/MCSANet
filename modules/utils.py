import numpy as np
import cv2
import torch
import torch.nn.functional as F
import argparse
import torch.distributed as dist

def penalty_builder(penalty_config):
    if penalty_config == '':
        return lambda x, y: y
    pen_type, alpha = penalty_config.split('_')
    alpha = float(alpha)
    if pen_type == 'wu':
        return lambda x, y: length_wu(x, y, alpha)
    if pen_type == 'avg':
        return lambda x, y: length_average(x, y, alpha)


def length_wu(length, logprobs, alpha=0.):
    """
    NMT length re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`.
    """

    modifier = (((5 + length) ** alpha) /
                ((5 + 1) ** alpha))
    return logprobs / modifier


def length_average(length, logprobs, alpha=0.):
    """
    Returns the average probability of tokens in a sequence.
    """
    return logprobs / length


def split_tensors(n, x):
    if torch.is_tensor(x):
        assert x.shape[0] % n == 0
        x = x.reshape(x.shape[0] // n, n, *x.shape[1:]).unbind(1)
    elif type(x) is list or type(x) is tuple:
        x = [split_tensors(n, _) for _ in x]
    elif x is None:
        x = [None] * n
    return x


def repeat_tensors(n, x):
    """
    For a tensor of size Bx..., we repeat it n times, and make it Bnx...
    For collections, do nested repeat
    """
    if torch.is_tensor(x):
        x = x.unsqueeze(1)  # Bx1x...
        x = x.expand(-1, n, *([-1] * len(x.shape[2:])))  # Bxnx...
        x = x.reshape(x.shape[0] * n, *x.shape[2:])  # Bnx...
    elif type(x) is list or type(x) is tuple:
        x = [repeat_tensors(n, _) for _ in x]
    return x


def generate_heatmap(image, weights):
    image = image.transpose(1, 2, 0)
    height, width, _ = image.shape
    weights = weights.reshape(int(weights.shape[0] ** 0.5), int(weights.shape[0] ** 0.5))
    weights = weights - np.min(weights)
    weights = weights / np.max(weights)
    weights = cv2.resize(weights, (width, height))
    weights = np.uint8(255 * weights)
    heatmap = cv2.applyColorMap(weights, cv2.COLORMAP_JET)
    result = heatmap * 0.5 + image * 0.5
    return result

def generate_heatmap_iu_xray(images, weights):
    batch_size, channels, height, width = images.shape
    # Reshape the weights into two parts, each part for one image
    half_len = len(weights) // 2
    weights1 = weights[:half_len]
    weights2 = weights[half_len:]
    # Process each image and its corresponding weights
    heatmaps = []
    for image, weights in zip(images, [weights1, weights2]):
        # Reshape weights to the height and width of the image
        weights = weights.reshape(int(weights.shape[0] ** 0.5), int(weights.shape[0] ** 0.5))
        # Normalize the weights
        weights = weights - np.min(weights)
        weights = weights / np.max(weights)
        # Resize weights to match the image dimensions
        weights = cv2.resize(weights, (width, height))
        weights = np.uint8(255 * weights)
        # Transpose the image to (H, W, C)
        image = image.transpose(1, 2, 0)
        # Apply color map to the weights
        heatmap = cv2.applyColorMap(weights, cv2.COLORMAP_JET)
        # Combine heatmap with the original image
        result = heatmap * 0.5 + image * 0.5
        # Append the result to the heatmaps list
        heatmaps.append(result)
    return heatmaps

def my_con_loss(features, labels, margin = 0.4, alpha = 1.5):
    B, _ = features.shape



    features = F.normalize(features)
    cos_matrix = features.mm(features.t())
    losses = []
    ortho_losses = []

    for i in range(B):
        max_sim = labels[i].new_ones(B)
        pos_label_matrix = labels[:, labels[i] == 1]
        pos_label_matrix = torch.sum(pos_label_matrix, dim=1)
        pos_label_matrix[pos_label_matrix != 0] = 1
        # 完全相同的标签
        exact_match = torch.all(labels[i] == labels, dim=1)
        max_sim[exact_match] = 1

        # 至少有一个标签相同但不是完全相同
        partial_match = (pos_label_matrix == 1) & ~exact_match
        label_diff = abs(labels[i] - labels[partial_match]).sum(dim=1)
        label_sum = (labels[i] + labels[partial_match]).sum(dim=1)
        max_sim[partial_match] = 1 / (alpha ** (label_diff / label_sum))


        # label_diff = abs(labels[i] - labels[pos_label_matrix == 1]).sum(dim = 1)
        # label_sum = (labels[i] + labels[pos_label_matrix == 1]).sum(dim=1)
        # max_sim[pos_label_matrix == 1] = 1/(alpha ** (label_diff/label_sum))

        neg_label_matrix = 1 - pos_label_matrix
        pos_cos_matrix = max_sim - cos_matrix[i, :]
        pos_cos_matrix[pos_cos_matrix < 0] = 0
        neg_cos_matrix = cos_matrix[i, :] - margin
        neg_cos_matrix[neg_cos_matrix < 0] = 0
        losses.append((pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum())

    loss = losses[0]
    for i in range(1, len(losses)):
        loss = loss + losses[i]
    loss /= (B * B)
    return loss



def cosine_similarity(x1, x2):
    return torch.sum(x1 * x2, dim=1)


def my_global_loss2(F_I, F_R, labels, alpha=0.7):
    B, D = F_I.shape
    # 初始化一个列表来存储所有损失
    losses = []
    F_I = F.normalize(F_I)
    F_R = F.normalize(F_R)

    for i in range(B):
        # 计算正样本对的余弦相似度
        sim_ij = cosine_similarity(F_I[i].unsqueeze(0), F_R[i].unsqueeze(0))

        # 创建负样本掩码：与 F_I[i] 标签完全不同的样本
        pos_label_matrix = labels[:, labels[i] == 1]
        pos_label_matrix = torch.sum(pos_label_matrix, dim=1)
        pos_label_matrix[pos_label_matrix != 0] = 1
        neg_label_matrix = (1 - pos_label_matrix).bool()

        # 选择硬负样本
        if neg_label_matrix.sum() > 0:
            # 计算 F_I[i] 和所有满足负样本掩码条件的 F_R 之间的余弦相似度，并取最大值
            sim_i_neg = cosine_similarity(F_I[i].unsqueeze(0), F_R[neg_label_matrix]).max()
        else:
            sim_i_neg = torch.tensor(0.0)  # 如果没有负样本，设置一个默认值

        # 计算损失分量
        loss = F.relu(alpha - sim_ij + sim_i_neg)
        # 将损失加入列表
        losses.append(loss)

    # 计算总损失
    total_loss = torch.stack(losses).mean()

    return total_loss

def global_loss_ir(F_I, F_R, labels):
    loss1 = my_global_loss2(F_I, F_R, labels)
    loss2 = my_global_loss2(F_R, F_I, labels)
    loss = loss1 + loss2
    return loss