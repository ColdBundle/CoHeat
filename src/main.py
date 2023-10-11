import pprint
import random

import click
import numpy as np
import torch
import torch.optim as optim
import yaml
from tqdm import tqdm

from model import CoHeat
from utils import Datasets


@click.command()
@click.option('--seed', type=int, default=0)
@click.option('--data', type=str, default='NetEase')
def main(seed, data):
    set_seed(seed)
    conf = yaml.safe_load(open("config.yaml"))[data]
    conf['dataset'] = data
    dataset = Datasets(conf)
    conf['num_users'] = dataset.num_users
    conf['num_bundles'] = dataset.num_bundles
    conf['num_items'] = dataset.num_items
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    conf['device'] = device
    pp = pprint.PrettyPrinter(indent=4, sort_dicts=False)
    pp.pprint(conf)

    model = CoHeat(conf, dataset.graphs, dataset.bundles_freq).to(device)
    optimizer = optim.Adam(model.parameters(), lr=conf["lr"], weight_decay=conf["lambda2"])
    crit = 20
    best_vld_rec, best_vld_ndcg, best_content = 0., 0., ''

    for epoch in range(1, conf["epochs"]+1):
        model.train(True)
        pbar = tqdm(enumerate(dataset.train_loader), total=len(dataset.train_loader))
        cur_instance_num, loss_avg, bpr_loss_avg, c_loss_avg = 0., 0., 0., 0.
        mult = epoch / conf["epochs"]
        psi = conf["max_temp"] ** mult

        for batch_i, batch in pbar:
            model.train(True)
            optimizer.zero_grad()
            batch = [x.to(device) for x in batch]

            bpr_loss, c_loss = model(batch, ED_drop=True, psi=psi)
            loss = bpr_loss + conf["lambda1"] * c_loss
            loss.backward()
            optimizer.step()

            loss_scalar = loss.detach()
            bpr_loss_scalar = bpr_loss.detach()
            c_loss_scalar = c_loss.detach()

            loss_avg = moving_avg(loss_avg, cur_instance_num, loss_scalar, batch[0].size(0))
            bpr_loss_avg = moving_avg(bpr_loss_avg, cur_instance_num, bpr_loss_scalar, batch[0].size(0))
            c_loss_avg = moving_avg(c_loss_avg, cur_instance_num, c_loss_scalar, batch[0].size(0))
            cur_instance_num += batch[0].size(0)
            pbar.set_description(f'epoch: {epoch:3d} | loss: {loss_avg:8.4f} | bpr_loss: {bpr_loss_avg:8.4f} | c_loss: {c_loss_avg:8.4f}')

        if epoch % conf['test_interval'] == 0:
            metrics = {}
            metrics['val'] = test(model, dataset.val_loader, conf, psi)
            metrics['test'] = test(model, dataset.test_loader, conf, psi)
            content = form_content(epoch, metrics['val'], metrics['test'], conf['topk'])
            print(content)
            if metrics['val']['recall'][crit] > best_vld_rec and metrics['val']['ndcg'][crit] > best_vld_ndcg:
                best_vld_rec = metrics['val']['recall'][crit]
                best_vld_ndcg = metrics['val']['ndcg'][crit]
                best_content = content

    print('============================ BEST ============================')
    print(best_content)


def set_seed(seed):
    """
    Set random seeds
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def moving_avg(avg, cur_num, add_value_avg, add_num):
    """
    Compute moving average
    """
    avg = (avg * cur_num + add_value_avg * add_num) / (cur_num + add_num)
    return avg


def form_content(epoch, val_results, test_results, ks):
    """
    Form a printing content
    """
    content = f'     Epoch|  Rec@{ks[0]} |  Rec@{ks[1]} |  Rec@{ks[2]} |  Rec@{ks[3]} |' \
             f' nDCG@{ks[0]} | nDCG@{ks[1]} | nDCG@{ks[2]} | nDCG@{ks[3]} |\n'
    val_content = f'{epoch:10d}|'
    val_results_recall = val_results['recall']
    for k in ks:
        val_content += f'  {val_results_recall[k]:.4f} |'
    val_results_ndcg = val_results['ndcg']
    for k in ks:
        val_content += f'  {val_results_ndcg[k]:.4f} |'
    content += val_content + '\n'
    test_content = f'{epoch:10d}|'
    test_results_recall = test_results['recall']
    for k in ks:
        test_content += f'  {test_results_recall[k]:.4f} |'
    test_results_ndcg = test_results['ndcg']
    for k in ks:
        test_content += f'  {test_results_ndcg[k]:.4f} |'
    content += test_content
    return content


def test(model, dataloader, conf, tau):
    """
    Test the model
    """
    tmp_metrics = {}
    for m in ["recall", "ndcg"]:
        tmp_metrics[m] = {}
        for topk in conf["topk"]:
            tmp_metrics[m][topk] = [0, 0]

    device = conf["device"]
    model.eval()
    rs = model.propagate(test=True)
    for users, ground_truth_u_b, train_mask_u_b in dataloader:
        pred_b = model.evaluate(rs, users.to(device), tau)
        pred_b -= 1e8 * train_mask_u_b.to(device)
        tmp_metrics = get_metrics(tmp_metrics, ground_truth_u_b, pred_b, conf["topk"])

    metrics = {}
    for m, topk_res in tmp_metrics.items():
        metrics[m] = {}
        for topk, res in topk_res.items():
            metrics[m][topk] = res[0] / res[1]

    return metrics


def get_metrics(metrics, grd, pred, topks):
    """
    Get recall and ndcg
    """
    tmp = {"recall": {}, "ndcg": {}}
    for topk in topks:
        _, col_indice = torch.topk(pred, topk)
        row_indice = torch.zeros_like(col_indice) + torch.arange(pred.shape[0], device=pred.device, dtype=torch.long).view(-1, 1)
        is_hit = grd[row_indice.view(-1).to('cpu'), col_indice.view(-1).to('cpu')].view(-1, topk)

        tmp["recall"][topk] = get_recall(pred, grd, is_hit, topk)
        tmp["ndcg"][topk] = get_ndcg(pred, grd, is_hit, topk)

    for m, topk_res in tmp.items():
        for topk, res in topk_res.items():
            for i, x in enumerate(res):
                metrics[m][topk][i] += x

    return metrics


def get_recall(pred, grd, is_hit, topk):
    """
    Get recall
    """
    epsilon = 1e-8
    hit_cnt = is_hit.sum(dim=1)
    num_pos = grd.sum(dim=1)

    # remove those test cases who don't have any positive items
    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = (hit_cnt/(num_pos+epsilon)).sum().item()

    return [nomina, denorm]


def get_ndcg(pred, grd, is_hit, topk):
    """
    Get ndcg
    """
    def DCG(hit, topk, device):
        hit = hit/torch.log2(torch.arange(2, topk+2, device=device, dtype=torch.float))
        return hit.sum(-1)

    def IDCG(num_pos, topk, device):
        hit = torch.zeros(topk, dtype=torch.float)
        hit[:num_pos] = 1
        return DCG(hit, topk, device)

    device = grd.device
    IDCGs = torch.empty(1+topk, dtype=torch.float)
    IDCGs[0] = 1  # avoid 0/0
    for i in range(1, topk+1):
        IDCGs[i] = IDCG(i, topk, device)

    num_pos = grd.sum(dim=1).clamp(0, topk).to(torch.long)
    dcg = DCG(is_hit, topk, device)

    idcg = IDCGs[num_pos]
    ndcg = dcg/idcg.to(device)

    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = ndcg.sum().item()

    return [nomina, denorm]


if __name__ == "__main__":
    main()
