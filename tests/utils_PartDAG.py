import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import GPy
import igraph as ig
import pandas as pd
import uuid
import math
import os
from torch.distributions import MultivariateNormal, Normal, Laplace, Gumbel
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel

from cdt.utils.R import RPackages, launch_R_script
from cdt.metrics import retrieve_adjacency_matrix

import networkx as nx
import json


class Dist(object):
    def __init__(self, d, noise_std = 2.5, noise_type = 'Gauss', adjacency = None, GP = True, lengthscale = 1, f_magn = 1, GraNDAG_like = False):
        self.d = d
        if isinstance(noise_std, (int, float)):
            noise_std = noise_std * torch.ones(self.d)
        self.GP = GP
        self.lengthscale = lengthscale
        self.f_magn = f_magn
        self.GraNDAG_like = GraNDAG_like

        
        if self.GraNDAG_like:
            noise_std = torch.ones(d)
        
        if noise_type == 'Gauss':
            self.noise = Normal(0, noise_std) # give standard deviation
        elif noise_type == 'Laplace':
            self.noise = Laplace(0, noise_std / np.sqrt(2))
        elif noise_type == 'Gumbel':
            self.noise = Gumbel(0, np.sqrt(noise_std) * np.sqrt(6)/np.pi)
        else:
            raise NotImplementedError("Unknown noise type.")
        
        self.adjacency = adjacency
        if adjacency is None:
            self.adjacency = np.ones((d,d))
            self.adjacency[np.tril_indices(d)] = 0

        # Needs strictly upper triangular matrix

        assert(np.allclose(self.adjacency, np.triu(self.adjacency)))


    def sampleGP(self, X, lengthscale=1):
        ker = GPy.kern.RBF(input_dim=X.shape[1],lengthscale=lengthscale,variance=self.f_magn)
        C = ker.K(X,X)
        X_sample = np.random.multivariate_normal(np.zeros(len(X)),C)
        return X_sample
    
    
    def sample_right(self, n):
        noise = self.noise.sample((n,)) # n x d noise matrix
        X = torch.zeros(n, self.d)
        node, Y = np.random.choice(self.d, 2, replace = False)
        

        # !!! Only works if adjacency matrix is upper triangular !!!
        noise_var = np.zeros(self.d)
        beta = []
        if self.GP:
            for i in range(self.d):
                if i == node:
                    parents = np.nonzero(self.adjacency[:,i])[0]
                    a = np.random.uniform(0, 1, len(parents))#.reshape([n, len(child[i])])
                    a[a > 0.5] = np.random.uniform(0.5, 2, sum(a > 0.5))
                    a[a < 0.5] = -np.random.uniform(0.5, 2, sum(a < 0.5))

                    np.random.shuffle(a)                    
                    if len(parents) > 0:
                        X_par = X[:,parents]
                        prob = torch.nn.Sigmoid()(torch.sum(torch.Tensor(a) * X_par, dim = 1))
                        X[:, i] = torch.Tensor(np.random.binomial(1, prob, n))
                    else:
                        X[:, i] = torch.Tensor(np.random.binomial(1, 0.5, n))

                else:
                    parents = np.nonzero(self.adjacency[:,i])[0]

                    a = np.random.uniform(0, 1, len(parents))
                    a[a > 0.5] = np.random.uniform(0.5, 2, sum(a > 0.5))
                    a[a < 0.5] = -np.random.uniform(0.5, 2, sum(a < 0.5))

                    np.random.shuffle(a)                    
                    if self.GraNDAG_like:
                        if len(parents) == 0: # For roots, noise variance sampled U(1,2)
                            noise_var[i] = np.random.uniform(1,2)
                        else: # Otherwise, noise variance sampled U(0.4,0.8)
                            noise_var[i] = np.random.uniform(0.4,0.8)
                        X[:, i] = np.sqrt(noise_var[i]) * noise[:,i]
                    else:
                        X[:, i] = noise[:,i]
                    if len(parents) > 0:
                        X_par = X[:,parents]
                        X[:, i] += torch.sum(torch.Tensor(a) * X_par, dim = 1)
                        
                        beta.append(a / np.array(torch.std(X[:, i])))
                        noise[:, i] = (noise[:, i] - np.array(torch.mean(X[:, i])))/ np.array(torch.std(X[:, i]))                        
                        X[:, i] = (X[:, i] - torch.mean(X[:, i]))/torch.std(X[:, i])

                    else:
                        beta.append([])
        else:
            for i in range(self.d):
                X[:, i] = noise[:,i]
                for j in np.nonzero(self.adjacency[:,i])[0]:
                    X[:, i] += torch.sin(X[:,j])
        return X, noise_var, noise, beta, Y, node    
        

    def log_p(self, X, active_nodes=None):
        if self.GP:
            raise NotImplementedError("Score computation not available with GPs.")
        if active_nodes is None:
            active_nodes = list(range(X.shape[1]))
        n = X.shape[0]
        d = X.shape[1]
        l = torch.zeros(n)
        for i, node_i in enumerate(active_nodes):
            fi = torch.zeros(n)
            for j, node_j in enumerate(active_nodes):
                if self.adjacency[node_j, node_i] != 0:
                    fi += torch.sin(X[:,j])
            l -= 0.5 * (X[:,i] - fi)**2
        return l

def sample_intervention_right(d, n, noise, beta, inter_value, node, Y, adj):

    X = noise.clone()

    flag = False
    for i in range(d):
        if i == node:
            X[:, i] = torch.Tensor(inter_value)

            flag = True
        else:
            parents = np.nonzero(adj[:,i])[0]

            if flag:
                a = beta[i - 1]
            else:
                a = beta[i]
            if len(parents) > 0:
                X_par = X[:,parents]
                X[:, i] += torch.sum(torch.Tensor(a) * X_par, dim = 1)

    return X    

def simulate_dag(d, s0, graph_type, triu=False):
    """Simulate random DAG with some expected number of edges.
    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF

    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """
    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        if triu:
            return np.triu(B_und, k=1)
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=False)
        B_und = _graph_to_adjmat(G)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'BP':
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    else:
        raise ValueError('unknown graph type')
    if not triu:
        B = _random_permutation(B)
    assert ig.Graph.Adjacency(B.tolist()).is_dag()
    return B


def full_DAG(top_order):
    d = len(top_order)
    A = np.zeros((d,d))
    for i, var in enumerate(top_order):
        A[var, top_order[i+1:]] = 1
    return A


def np_to_csv(array, save_path):
    """
    Convert np array to .csv
    array: numpy array
        the numpy array to convert to csv
    save_path: str
        where to temporarily save the csv
    Return the path to the csv file
    """
    id = str(uuid.uuid4())
    output = os.path.join(os.path.dirname(save_path), 'tmp_' + id + '.csv')

    df = pd.DataFrame(array)
    df.to_csv(output, header=False, index=False)

    return output



def pns_(model_adj, x, num_neighbors, thresh):
    """Preliminary neighborhood selection"""
    num_samples = x.shape[0]
    num_nodes = x.shape[1]
    print("PNS: num samples = {}, num nodes = {}".format(num_samples, num_nodes))
    for node in range(num_nodes):
        print("PNS: node " + str(node))
        x_other = np.copy(x)
        x_other[:, node] = 0
        reg = ExtraTreesRegressor(n_estimators=500)
        reg = reg.fit(x_other, x[:, node])
        selected_reg = SelectFromModel(reg, threshold="{}*mean".format(thresh), prefit=True,
                                       max_features=num_neighbors)
        mask_selected = selected_reg.get_support(indices=False).astype(np.float)

        model_adj[:, node] *= mask_selected

    return model_adj

def edge_errors(pred, target):
    """
    Counts all types of edge errors (false negatives, false positives, reversed edges)
    """
    true_labels = retrieve_adjacency_matrix(target)
    predictions = retrieve_adjacency_matrix(pred, target.nodes() if isinstance(target, nx.DiGraph) else None)

    diff = true_labels - predictions

    rev = (((diff + diff.transpose()) == 0) & (diff != 0)).sum() / 2
    # Each reversed edge necessarily leads to one fp and one fn so we need to subtract those
    fn = (diff == 1).sum() - rev
    fp = (diff == -1).sum() - rev

    return fn, fp, rev


def SHD(pred, target):
    return sum(edge_errors(pred, target))


class MLP(nn.Module):
    def __init__(self, X, T):
        super(MLP, self).__init__() 
        self.X = X
        self.T = T
        self.linear_1 = torch.nn.Linear(X.shape[1], 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, x):
        return self.sigmoid(self.linear_1(x)).squeeze()

    def fit(self, num_epoch=1000, lr=0.01, lamb=0, tol=1e-4, batch_size=10, verbose=0):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        num_sample = len(self.X)
        total_batch = num_sample // batch_size

        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0
            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = torch.Tensor(self.X[selected_idx]).cuda()
                sub_t = torch.Tensor(self.T[selected_idx]).cuda()

                optimizer.zero_grad()
                pred = self.forward(sub_x)
                
                xent_loss = nn.BCELoss()(pred, sub_t)

                loss = xent_loss
                loss.backward()
                optimizer.step()
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MLP] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MLP] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[Warning] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(torch.Tensor(x).cuda())
        return pred.detach().cpu().numpy().flatten()    
    
    
class LR(nn.Module):
    def __init__(self, X, flag = False):
        super(LR, self).__init__() 
        self.X = X   
        self.linear_1 = torch.nn.Linear(self.X.shape[1], 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.flag = flag
        
    def forward(self, x, flag = False):
        if flag:
            return self.sigmoid(self.linear_1(x).squeeze())
        else:
            return self.linear_1(x).squeeze()
    def fit(self, Y, num_epoch=1000, lr=0.01, lamb=1e-4, tol=1e-4, batch_size=10, verbose=0):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        num_sample = len(self.X)
        total_batch = num_sample // batch_size

        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0
            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = torch.Tensor(self.X[selected_idx]).cuda()
                sub_y = torch.Tensor(Y[selected_idx]).cuda()
                
                optimizer.zero_grad()
                pred = self.forward(sub_x, self.flag)

                xent_loss = nn.MSELoss()(pred, sub_y)

                loss = xent_loss
                loss.backward()
                optimizer.step()
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MLP] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MLP] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[Warning] Reach preset epochs, it seems does not converge.")
    
    def predict(self, x, flag = False):
        pred = self.forward(torch.Tensor(x).cuda(), flag)
        return pred.detach().cpu().numpy().flatten()
    
    
class IPS(nn.Module):
    def __init__(self, X, output_dim):
        super(IPS, self).__init__() 
        self.X = X
        self.linear_1 = torch.nn.Linear(X.shape[1], output_dim)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.linear_1(x).squeeze()
    
    def update_y(self, T, Y, weight):
        return T * Y/weight - (1 - T) * Y/(1 - weight)
    
    def predict(self, x):
        pred = self.forward(torch.Tensor(x).cuda())
        return pred.detach().cpu().numpy().flatten()
           
class Model_Multi(nn.Module):
    def __init__(self, X, Y, T, propensity, output_dim, flag = False):
        super(Model_Multi, self).__init__() 
        self.X = X
        self.Y = Y
        self.T = T
        self.LR = LR(X, flag)
        self.IPS = IPS(X, output_dim)
        self.propensity = propensity
        self.flag = flag

    def fit(self, constrain = 10, C = 0.2, alpha = 1, beta = 1, num_epoch=1000, lr=0.01, lamb=1e-4, tol=1e-4, batch_size=10, verbose=0):
        optimizer_pred = torch.optim.Adam(self.LR.parameters(), lr=lr, weight_decay=lamb)
        optimizer_cate = torch.optim.Adam(self.IPS.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9

        num_sample = len(self.X)
        total_batch = num_sample // batch_size

        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0
            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = torch.Tensor(self.X[selected_idx]).cuda()
                sub_y = torch.Tensor(self.Y[selected_idx]).cuda()               
                sub_t = torch.Tensor(self.T[selected_idx]).cuda()
                
                pred = self.LR.forward(sub_x, self.flag)

                xent_loss = nn.MSELoss()(pred, sub_y)

                pred = pred.reshape(batch_size, -1)

                CATE_bound = torch.max(torch.sum(torch.clip(-C-self.IPS.forward(sub_x).reshape(batch_size, -1), 0, 100) + torch.clip(-C+self.IPS.forward(sub_x).reshape(batch_size, -1), 0, 100), dim = 0))

                sub_propensity = torch.Tensor(self.propensity[selected_idx, :]).cuda() # propensity

                CATE_y = self.IPS.update_y(sub_t.reshape(batch_size, -1), pred, sub_propensity.reshape(batch_size, -1))
    
                pred_cate = self.IPS.forward(sub_x).reshape(batch_size, -1)

                loss = beta * torch.mean((pred_cate - CATE_y) ** 2) + alpha * xent_loss + constrain * CATE_bound

                optimizer_pred.zero_grad()
                optimizer_cate.zero_grad()
                loss.backward()
                optimizer_pred.step()
                optimizer_cate.step()
                epoch_loss += loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[LR] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[LR] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[Warning] Reach preset epochs, it seems does not converge.")
                