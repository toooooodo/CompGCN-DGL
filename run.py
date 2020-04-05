import argparse
import time
from pprint import pprint
import numpy as np
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from dgl.contrib.data import load_data

from model import CompGCN_DistMult, CompGCN_ConvE
from utils import process, TrainDataset, TestDataset


class Runner(object):
    def __init__(self, params):
        self.p = params
        self.prj_path = Path(__file__).parent.resolve()
        self.time_stamp = time.strftime('%Y_%m_%d') + '_' + time.strftime('%H:%M:%S')
        self.data = load_data(self.p.dataset)
        self.num_ent, self.train_data, self.valid_data, self.test_data, self.num_rels = self.data.num_nodes, self.data.train, self.data.valid, self.data.test, self.data.num_rels
        self.triplets = process({'train': self.train_data, 'valid': self.valid_data, 'test': self.test_data},
                                self.num_ent)
        if self.p.gpu != -1 and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.p.gpu}')
            # -------------------------------
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.deterministic = True
            # -------------------------------
        else:
            self.device = torch.device('cpu')
        self.p.embed_dim = self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim  # output dim of gnn
        self.data_iter = self.get_data_iter()
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.p.lr, weight_decay=self.p.l2)

    def fit(self):
        pass

    def get_data_iter(self):
        """
        get data loader for train, valid and test section
        :return: dict
        """
        def get_data_loader(dataset_class, split):
            return DataLoader(
                dataset_class(self.triplets[split], self.num_ent, self.p),
                batch_size=self.p.batch_size,
                shuffle=True,
                num_workers=self.p.num_workers
            )

        return {
            'train': get_data_loader(TrainDataset, 'train'),
            'valid_head': get_data_loader(TestDataset, 'valid_head'),
            'valid_tail': get_data_loader(TestDataset, 'valid_tail'),
            'test_head': get_data_loader(TestDataset, 'test_head'),
            'test_tail': get_data_loader(TestDataset, 'test_tail')
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--name', default='test_run', help='Set run name for saving/restoring models')
    parser.add_argument('--data', dest='dataset', default='FB15k-237', help='Dataset to use, default: FB15k-237')
    parser.add_argument('--model', dest='model', default='compgcn', help='Model Name')
    parser.add_argument('--score_func', dest='score_func', default='conve',
                        help='Score Function for Link prediction')
    parser.add_argument('--opn', dest='opn', default='mult', help='Composition Operation to be used in CompGCN')

    parser.add_argument('--batch', dest='batch_size', default=128, type=int, help='Batch size')
    # parser.add_argument('--gamma', type=float, default=40.0, help='Margin')
    parser.add_argument('--gpu', type=str, default='0', help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('--epoch', dest='max_epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--l2', type=float, default=0.0, help='L2 Regularization for Optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='Starting Learning Rate')
    parser.add_argument('--lbl_smooth', dest='lbl_smooth', type=float, default=0.1, help='Label Smoothing')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of processes to construct batches')
    parser.add_argument('--seed', dest='seed', default=12345, type=int, help='Seed for randomization')

    # parser.add_argument('--restore', dest='restore', action='store_true',
    #                     help='Restore from the previously saved model')
    parser.add_argument('--bias', dest='bias', action='store_true', help='Whether to use bias in the model')

    parser.add_argument('--num_bases', dest='num_bases', default=-1, type=int,
                        help='Number of basis relation vectors to use')
    parser.add_argument('--init_dim', dest='init_dim', default=100, type=int,
                        help='Initial dimension size for entities and relations')
    parser.add_argument('--gcn_dim', dest='gcn_dim', default=200, type=int, help='Number of hidden units in GCN')
    parser.add_argument('--embed_dim', dest='embed_dim', default=None, type=int,
                        help='Embedding dimension to give as input to score function')
    parser.add_argument('--gcn_layer', dest='gcn_layer', default=1, type=int, help='Number of GCN Layers to use')
    parser.add_argument('--gcn_drop', dest='dropout', default=0.1, type=float, help='Dropout to use in GCN Layer')
    parser.add_argument('--hid_drop', dest='hid_drop', default=0.3, type=float, help='Dropout after GCN')

    # ConvE specific hyperparameters
    parser.add_argument('--conve_hid_drop', dest='conve_hid_drop', default=0.3, type=float,
                        help='ConvE: Hidden dropout')
    parser.add_argument('--feat_drop', dest='feat_drop', default=0.2, type=float, help='ConvE: Feature Dropout')
    parser.add_argument('--input_drop', dest='input_drop', default=0.2, type=float, help='ConvE: Stacked Input Dropout')
    parser.add_argument('--k_w', dest='k_w', default=20, type=int, help='ConvE: k_w')
    parser.add_argument('--k_h', dest='k_h', default=10, type=int, help='ConvE: k_h')
    parser.add_argument('--num_filt', dest='num_filt', default=200, type=int,
                        help='ConvE: Number of filters in convolution')
    parser.add_argument('--ker_sz', dest='ker_sz', default=7, type=int, help='ConvE: Kernel size to use')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    runner = Runner(args)
    runner.fit()
