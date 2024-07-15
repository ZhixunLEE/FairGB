import argparse
import torch
import numpy as np
import torch.nn.functional as F

from models import *
from data_utils import get_dataset
from eval import evaluate_ged3
from utils import seed_everything, get_enc_cls_opt
from mixup import sampling_idx_individual_dst, neighbor_sampling, get_ins_neighbor_dist, saliency_mixup

EPS = 1e-6

def run(data, args):
    acc, f1, auc_roc, parity, equality = np.zeros(args.runs), np.zeros(
        args.runs), np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs)
    
    neighbor_dist_list = get_ins_neighbor_dist(data.y.size(0), data.edge_index, args.device)

    data = data.to(args.device)
    n_cls = data.y.max().int().item() + 1
    n_sen = data.sens.max().int().item() + 1
    index_list = torch.arange(len(data.y)).to(args.device)
    group_num_list, idx_info = [], []
    for i in range(n_cls):
        for j in range(n_sen):
            mask = ((data.y == i) & (data.sens == j) & data.train_mask)
            data_num = mask.sum()
            group_num_list.append(int(data_num.item()))
            idx_info.append(index_list[mask])

    encoder, classifier, optimizer_e, optimizer_c = get_enc_cls_opt(args)

    for count in range(args.runs):
        seed_everything(count + args.seed)
        classifier.reset_parameters()
        encoder.reset_parameters()

        best_val_tradeoff = 0
        for epoch in range(0, args.epochs):
            encoder.train()
            classifier.train()

            optimizer_c.zero_grad()
            optimizer_e.zero_grad()
            
            sampling_src_idx, sampling_dst_idx = sampling_idx_individual_dst(group_num_list, idx_info, args.eta)
            beta = torch.distributions.beta.Beta(2, 2)
            lam = beta.sample((len(sampling_src_idx),) ).unsqueeze(1)
            lam = lam.to(args.device)
        
            if epoch >= args.warmup:
                new_edge_index = neighbor_sampling(data.x.size(0), data.edge_index, sampling_src_idx, neighbor_dist_list)
                new_x = saliency_mixup(data.x, sampling_src_idx, sampling_dst_idx, lam)
                
                h = encoder(new_x, new_edge_index)
                output = classifier(h)

                add_num = output.shape[0] - data.train_mask.shape[0]
                new_train_mask = torch.ones(add_num, dtype=torch.bool, device=args.device)
                new_train_mask = torch.cat((torch.zeros(data.train_mask.shape[0], dtype=torch.bool, device=args.device), new_train_mask), dim=0)

                loss_src = F.binary_cross_entropy_with_logits(
                    output[new_train_mask], data.y[sampling_src_idx].unsqueeze(1).to(args.device), reduction='none')
                loss_dst = F.binary_cross_entropy_with_logits(
                    output[new_train_mask], data.y[sampling_dst_idx].unsqueeze(1).to(args.device), reduction='none')
                
                pos_grad_src = (1. - torch.exp(-loss_src).detach()) * lam
                pos_grad_dst = (1. - torch.exp(-loss_dst).detach()) * (1-lam)
                grad_count = []
                for i in range(n_cls):
                    for j in range(n_sen):
                        mask_src = (data.y[sampling_src_idx] == i) & (data.sens[sampling_src_idx] == j)
                        mask_dst = (data.y[sampling_dst_idx] == i) & (data.sens[sampling_dst_idx] == j)
                        grad_count.append(pos_grad_src[mask_src].sum().item() + pos_grad_dst[mask_dst].sum().item())

                min_grad = np.min(grad_count)
                group_weight_list = [float(min_grad)/(float(num) + EPS) for num in grad_count]

                for i in range(n_cls):
                    for j in range(n_sen):
                        mask_src = (data.y[sampling_src_idx] == i) & (data.sens[sampling_dst_idx] == j)
                        mask_dst = (data.y[sampling_dst_idx] == i) & (data.sens[sampling_dst_idx] == j)
                        loss_src[mask_src] *= group_weight_list[i*2+j]
                        loss_dst[mask_dst] *= group_weight_list[i*2+j]

                loss = lam * loss_src + (1-lam) * loss_dst
                loss.mean().backward()
            else:
                h = encoder(data.x, data.edge_index)
                output = classifier(h)

                loss_c = F.binary_cross_entropy_with_logits(
                    output[data.train_mask], data.y[data.train_mask].unsqueeze(1).to(args.device))
                loss_c.backward()

            optimizer_e.step()
            optimizer_c.step()

            accs, auc_rocs, F1s, tmp_parity, tmp_equality = evaluate_ged3(classifier, encoder, data)

            if epoch % 10 == 0:
                print("RUN: {}/{}, Epoch: {:04}/{:04} | Val Acc: {:.4f}, Test Acc: {:.4f}, Test AUC: {:.4f}, Test F1: {:.4f}, Test SP: {:.4f}, Test EO: {:.4f}".format(
                    count+1, args.runs, epoch, args.epochs, accs['val'], accs['test'], auc_rocs['test'], F1s['test'], tmp_parity['test'], tmp_equality['test']
                ))

            if (auc_rocs['val'] + F1s['val'] + accs['val'] - args.alpha * (tmp_parity['val'] + tmp_equality['val']) > best_val_tradeoff):
                test_acc = accs['test']
                test_auc_roc = auc_rocs['test']
                test_f1 = F1s['test']
                test_parity, test_equality = tmp_parity['test'], tmp_equality['test']

                best_val_tradeoff = auc_rocs['val'] + F1s['val'] + \
                    accs['val'] - args.alpha * (tmp_parity['val'] + tmp_equality['val'])
                                
                print("\033[0;30;41m RUN: {}/{}, Epoch: {:04}/{:04} | Val Acc: {:.4f}, Test Acc: {:.4f}, Test AUC: {:.4f}, Test F1: {:.4f}, Test SP: {:.4f}, Test EO: {:.4f}\033[0m".format(
                    count+1, args.runs, epoch, args.epochs, accs['val'], accs['test'], auc_rocs['test'], F1s['test'], tmp_parity['test'], tmp_equality['test']
                ))

        acc[count] = test_acc
        f1[count] = test_f1
        auc_roc[count] = test_auc_roc
        parity[count] = test_parity
        equality[count] = test_equality

    return acc, f1, auc_roc, parity, equality


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='german')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--c_lr', type=float, default=0.01)
    parser.add_argument('--c_wd', type=float, default=0)
    parser.add_argument('--e_lr', type=float, default=0.01)
    parser.add_argument('--e_wd', type=float, default=0)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--encoder', type=str, default='SAGE')
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--eta', type=float, default=0.5)

    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu_num) if torch.cuda.is_available() else 'cpu')
    data, args.sens_idx, args.x_min, args.x_max = get_dataset(args.dataset)
    args.num_features, args.num_classes = data.x.shape[1], 1

    acc, f1, auc_roc, parity, equality  = run(data, args)
    print('======' + args.dataset + args.encoder + '======')
    print('auc_roc: {:.2f} +- {:.2f}'.format(np.mean(auc_roc) * 100, np.std(auc_roc) * 100))
    print('Acc: {:.2f} +- {:.2f}'.format(np.mean(acc) * 100, np.std(acc) * 100))
    print('f1: {:.2f} +- {:.2f}'.format(np.mean(f1) * 100, np.std(f1) * 100))
    print('parity: {:.2f} +- {:.2f}'.format(np.mean(parity) * 100, np.std(parity) * 100))
    print('equality: {:.2f} +- {:.2f}'.format(np.mean(equality) * 100, np.std(equality) * 100))