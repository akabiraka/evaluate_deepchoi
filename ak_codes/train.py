import sys
sys.path.append("../deepchoi")

import torch, gc
import torch.nn as nn
import argparse
import numpy as np
import pickle
import time
from Utils import FC, Embedder, GCN
import Making_Graph
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

gc.collect()
torch.cuda.empty_cache()

import pickle_utils
from GO import Ontology
import os.path as osp
import scipy.sparse as sp
from ak_codes.val import run_val
from ak_codes.test import GOname, run_test
from config import Config

"""To save time and money, we included the results from SeqVec in the training dataset in advance."""

class CustomDataset(Dataset):
    def __init__(self, species, GO, data_generation_process, dataset="train") -> None:
        super(CustomDataset, self).__init__()
        self.species = species
        self.GO = GO
        self.data_generation_process = data_generation_process
        
        self.dataset_annots = pickle_utils.load_pickle(f"ak_data/{data_generation_process}/{GO}/{dataset}.pkl")
        self.terms_dict = pickle_utils.load_pickle(f"ak_data/{data_generation_process}/{GO}/studied_terms.pkl")
        self.studied_terms_set = set(self.terms_dict.keys())

        self.train_dataset = pickle_utils.load_pickle(f"ak_data/{self.data_generation_process}/{self.GO}/train.pkl") # list of uniprot_id, set([terms])
        self.train_annots = [annots for uniprot_id, annots in self.train_dataset]
        self.GOid_vs_uniprotids_dict = self.terms_annotated_to()

        self.go_rels = Ontology('ak_data/go.obo', with_rels=True)
        
    def __len__(self):
        return len(self.dataset_annots)


    def generate_true_label(self, annots):
        y_true = torch.zeros(len(self.terms_dict), dtype=torch.float32)
        for term in annots:
            y_true[self.terms_dict[term]] = 1.
        return y_true

    def get_seq_representation(self, uniprot_id):
        seq_rep = pickle_utils.load_pickle(f"ak_data/{self.species}_sequences_rep_mean/{uniprot_id}.pkl") # shape: [esm1b_embed_dim]
        return seq_rep

    def __getitem__(self, i):
        uniprot_id, annots = self.dataset_annots[i]
        
        y_true = self.generate_true_label(annots) # shape: [n_terms]
        seq_rep = self.get_seq_representation(uniprot_id) # shape: [esm1b_embed_dim]

        return seq_rep, y_true


    def terms_annotated_to(self):
        GOid_vs_uniprotids_dict = {}
        for term in self.terms_dict.keys():
            uniprotid_list = []
            for uniprot_id, annots in self.train_dataset:
                if term in annots:
                    uniprotid_list.append(uniprot_id)

            GOid_vs_uniprotids_dict[term] = set(uniprotid_list)
        return GOid_vs_uniprotids_dict


    def get_terms_rep(self): #  one-hot encoding matrix where the i th row GO term and its ancestors are 1
        terms_rep_path = f"ak_data/{self.data_generation_process}/{self.GO}/terms_rep.pkl"
        if osp.exists(terms_rep_path): 
            return pickle_utils.load_pickle(terms_rep_path)


        rep = np.zeros(shape=(len(self.terms_dict), len(self.terms_dict)), dtype=np.int16)
        np.fill_diagonal(rep, 1)

        for term, i in self.terms_dict.items():
            anchestor_terms = self.go_rels.get_anchestors(term)
            anchestor_terms = self.studied_terms_set.intersection(anchestor_terms)

            for anchestor_term in anchestor_terms:
                term_i = self.terms_dict.get(anchestor_term)
                rep[i, term_i] = 1
        rep = torch.tensor(rep, dtype=torch.float32)
        pickle_utils.save_as_pickle(rep, terms_rep_path)
        return rep


    def get_terms_adj(self):
        terms_adj_path = f"ak_data/{self.data_generation_process}/{self.GO}/terms_adj.pkl"
        if osp.exists(terms_adj_path): 
            return pickle_utils.load_pickle(terms_adj_path)

        self.go_rels.calculate_ic(self.train_annots)

        adj = np.zeros(shape=(len(self.terms_dict), len(self.terms_dict)), dtype=np.float32)
        np.fill_diagonal(adj, 1.0)
        
        for parent_term, i in self.terms_dict.items():
            child_terms = self.go_rels.get_children(parent_term)
            child_terms = self.studied_terms_set.intersection(child_terms)

            total_IC_of_children = 0.0
            for child_term in child_terms:
                total_IC_of_children += self.go_rels.get_ic(child_term)
            
            if total_IC_of_children <= 0.0: relative_IC = self.go_rels.get_ic(parent_term)
            else: relative_IC = self.go_rels.get_ic(parent_term) / total_IC_of_children
            
            N_t = len(self.GOid_vs_uniprotids_dict[parent_term])
            if N_t==0:
                print(self.GOid_vs_uniprotids_dict[parent_term])
            for child_term in child_terms:
                N_s = len(self.GOid_vs_uniprotids_dict[child_term])

                adj[self.terms_dict[parent_term], self.terms_dict[child_term]] = (N_s/N_t) + relative_IC
        
        adj = sp.coo_matrix(adj)
        adj = adj + np.multiply(adj.T, adj.T > adj) - np.multiply(adj, (adj.T > adj))
        adj = self.normalize(adj + sp.eye(adj.shape[0]))
        adj = self.sparse_mx_to_torch_sparse_tensor(adj)
        pickle_utils.save_as_pickle(adj, terms_adj_path)
        return adj


    def normalize(self, adj):
        rowsum = np.array(adj.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        adj = r_mat_inv.dot(adj)
        return adj


    def sparse_mx_to_torch_sparse_tensor(self, sparse_adj):
        sparse_adj = sparse_adj.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_adj.row, sparse_adj.col)).astype(np.int64))
        values = torch.from_numpy(sparse_adj.data)
        shape = torch.Size(sparse_adj.shape)
        return torch.sparse.LongTensor(indices, values, shape)




class Net(nn.Module):
    def __init__(self, n_class, args):
        super().__init__()
        self.FC = FC(768, args.seqfeat)
        self.Graph_Embedder = Embedder(n_class, args.nfeat)
        self.GCN = GCN(args.nfeat, args.nhid)

    def forward(self, seq, node, adj):
        seq_out = self.FC(seq)
        node_embd = self.Graph_Embedder(node)
        graph_out = self.GCN(node_embd, adj)
        graph_out = graph_out.transpose(-2, -1)
        output = torch.matmul(seq_out, graph_out)
        output = torch.sigmoid(output)
        return output

def eval_model(model, one_hot_node, adj, device, eval_loader):
    pred_scores = []
    for i, (input, target) in enumerate(eval_loader):
        # print(input.shape, target.shape)
        input, target = input.to(device), target.to(device)
        preds = model(input, one_hot_node, adj)
        pred_scores.append(preds.detach().cpu().numpy())
    pred_scores = np.vstack(pred_scores)
    return pred_scores

def train_model(args):
    c = Config()
    species, GOname, data_generation_process = c.species, c.GO, c.data_generation_process
    print("training model...")
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(device)

    tr_dataset = CustomDataset(species, GOname, data_generation_process, "train")
    n_classes = len(tr_dataset.terms_dict)
    # seq_rep, y_true = tr_dataset.__getitem__(0)
    # print(seq_rep.shape, y_true.shape)
    train_loader = DataLoader(dataset=tr_dataset, batch_size=args.batch_size)
    print(len(train_loader))


    one_hot_node = tr_dataset.get_terms_rep() 
    print(one_hot_node.shape)

    adj = tr_dataset.get_terms_adj()
    print(adj.shape)

    # Data load
    # adj, one_hot_node, label_map, label_map_ivs = Making_Graph.build_graph()
    # print(adj.shape, one_hot_node.shape, label_map.shape)
    

    # model definition
    model = Net(n_classes, args).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    

    model.train()
    total_loss = 0
    print_every = 5
    start = time.time()
    temp = start


    one_hot_node, adj = one_hot_node.to(device), adj.to(device)
    for epoch in range(args.epochs):
        train_loss = 0

        for i, (input, target) in enumerate(train_loader):
            # print(input.shape, target.shape)
            input, target = input.to(device), target.to(device)
            preds = model(input, one_hot_node, adj)

            optimizer.zero_grad()
            loss = criterion(preds, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            ## record the average training loss, using something like
            train_loss += loss.item()
            batch_idx = i
            if (i + 1) % print_every == 0:
                loss_avg = total_loss / print_every
                print("time = %dm, epoch %d, iter = %d, loss = %.3f,%ds per %d iters" % ((time.time() - start) // 60,
                                                                                         epoch + 1, i + 1, loss_avg,
                                                                                         time.time() - temp,
                                                                                         print_every))
                total_loss = 0
                temp = time.time()

        train_loss = train_loss / batch_idx
        print("batch_idx", i)
        print("train_loss", train_loss)


        # model save
    #     torch.save({
    #         'startEpoch': epoch + 1,
    #         'loss': train_loss,
    #         'state_dict': model.state_dict(),
    #         'optimizer': optimizer.state_dict()
    #     }, os.path.join("Weights/BPO", f'model_{epoch + 1:02d}.pth'))
    # torch.save(model, 'Weights/BPO/final.pth')

    model.eval()
    val_dataset = CustomDataset(species, GOname, data_generation_process, "val")
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    val_pred_scores = eval_model(model, one_hot_node, adj, device, val_loader)
    run_val(val_pred_scores)

    test_dataset = CustomDataset(species, GOname, data_generation_process, "test")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    test_pred_scores = eval_model(model, one_hot_node, adj, device, test_loader)
    run_test(test_pred_scores)


def main():
    parser = argparse.ArgumentParser()

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)

    # Model parameters
    """BPO has 80 parameters, MFO has 41 parameters, and CCO has 54 parameters"""
    parser.add_argument("--nfeat", type=int, default=80, help="node feature size")
    parser.add_argument("--nhid", type=int, default=80, help="GCN node hidden size")
    parser.add_argument("--seqfeat", type=int, default=80, help="sequence reduced feature size")

    args = parser.parse_args()

    train_model(args)


if __name__ == '__main__':
    main()
