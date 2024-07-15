import torch
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter_add


@torch.no_grad()
def sampling_idx_individual_dst(group_num_list, idx_info, eta=0.5):
    n_cls, n_grp = 2, 2
    sampling_src_idx = torch.cat(idx_info)
    if np.random.rand() < eta:
        inter = True
    else:
        inter = False
    sampling_dst_idx = []
    for i in range(n_cls):
        for j in range(n_grp):
            if inter:
                target_group_id = 2 * (1 - i) + j
            else:
                target_group_id = 2 * i + (1 - j)
            prob = torch.ones(group_num_list[target_group_id]) / group_num_list[target_group_id]
            sampled_idx = torch.multinomial(prob, group_num_list[i * 2 + j], replacement=True)
            sampled_idx = idx_info[target_group_id][sampled_idx]
            sampling_dst_idx.append(sampled_idx)
    
    sampling_dst_idx = torch.cat(sampling_dst_idx)
    
    sampling_src_idx, sorted_idx = torch.sort(sampling_src_idx)
    sampling_dst_idx = sampling_dst_idx[sorted_idx]

    return sampling_src_idx, sampling_dst_idx


def saliency_mixup(x, sampling_src_idx, sampling_dst_idx, lam):
    new_src = x[sampling_src_idx.to(x.device), :].clone()
    new_dst = x[sampling_dst_idx.to(x.device), :].clone()
    lam = lam.to(x.device)

    mixed_node = lam * new_src + (1-lam) * new_dst
    new_x = torch.cat([x, mixed_node], dim =0)
    return new_x


@torch.no_grad()
def neighbor_sampling(total_node, edge_index, sampling_src_idx, neighbor_dist_list):
    ## Exception Handling ##
    device = edge_index.device
    sampling_src_idx = sampling_src_idx.clone().to(device)

    # Find the nearest nodes and mix target pool
    mixed_neighbor_dist = neighbor_dist_list[sampling_src_idx]

    # Compute degree
    col = edge_index[1]
    degree = scatter_add(torch.ones_like(col), col)
    if len(degree) < total_node:
        degree = torch.cat([degree, degree.new_zeros(total_node-len(degree))],dim=0)
    train_node_mask = torch.ones_like(degree,dtype=torch.bool)
    degree_dist = scatter_add(torch.ones_like(degree[train_node_mask]), degree[train_node_mask]).to(device).type(torch.float32)

    # Sample degree for augmented nodes
    prob = degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx),1)
    aug_degree = torch.multinomial(prob, 1).to(device).squeeze(dim=1) # (m)
    max_degree = degree.max().item() + 1
    aug_degree = torch.min(aug_degree, degree[sampling_src_idx])

    # Sample neighbors
    new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, max_degree)
    tgt_index = torch.arange(max_degree).unsqueeze(dim=0).to(device)
    new_col = new_tgt[(tgt_index - aug_degree.unsqueeze(dim=1) < 0)]
    new_row = (torch.arange(len(sampling_src_idx)).to(device)+ total_node)
    new_row = new_row.repeat_interleave(aug_degree)
    inv_edge_index = torch.stack([new_col, new_row], dim=0)
    new_edge_index = torch.cat([edge_index, inv_edge_index], dim=1)

    return new_edge_index


from tqdm import tqdm

@torch.no_grad()
def get_ins_neighbor_dist(num_nodes, edge_index, device):
    """
    Compute adjacent node distribution.
    """
    ## Utilize GPU ##
    # edge_index = edge_index.clone().to(device)
    edge_index = edge_index.clone()
    row, col = edge_index[0], edge_index[1]

    # Compute neighbor distribution
    neighbor_dist_list = []
    for j in tqdm(range(num_nodes)):
        # neighbor_dist = torch.zeros(num_nodes, dtype=torch.float32).to(device)
        neighbor_dist = torch.zeros(num_nodes, dtype=torch.float32)

        idx = row[(col==j)]
        neighbor_dist[idx] = neighbor_dist[idx] + 1
        neighbor_dist_list.append(neighbor_dist)

    neighbor_dist_list = torch.stack(neighbor_dist_list,dim=0).to(device)
    neighbor_dist_list = F.normalize(neighbor_dist_list,dim=1,p=1)

    return neighbor_dist_list