import torch as t
from torch import nn
import torch.nn.functional as F
from Params import args
from Utils import contrastLoss, ce, l2_norm
from Utils import keras_style_init


class Model(nn.Module):
    def __init__(self, initializer='glorot_uniform'):
        super(Model, self).__init__()

        self.dEmbeds = nn.Parameter(keras_style_init(t.empty(args.drug, args.latdim), initializer))
        self.gEmbeds = nn.Parameter(keras_style_init(t.empty(args.gene, args.latdim), initializer))

        self.gatLite = GATLite(in_dim=args.latdim, out_dim=args.latdim)
        self.gatedGCN = GatedGCN(in_dim=args.latdim, hidden_dim=64, num_layers=2)

        self.gate = nn.Sequential(
            nn.Linear(args.latdim, args.latdim),
            nn.Sigmoid()
        )

        self.classifierLayer = ClassifierLayer(initializer)

        if args.dense:
            self.dHyper = nn.Parameter(keras_style_init(t.empty(args.latdim, args.hyperNum), initializer))
            self.gHyper = nn.Parameter(keras_style_init(t.empty(args.latdim, args.hyperNum), initializer))

        self.edgeDropper = SpAdjDropEdge()

    def forward(self, adj, keepRate):
        embeds = t.concat([self.dEmbeds, self.gEmbeds], axis=0)
        embedsLst = [embeds]
        gatEmbedsLst = [embeds]
        hyperEmbedsLst = [embeds]

        ddHyper = self.dEmbeds * args.mult
        ggHyper = self.gEmbeds * args.mult

        if args.dense:
            ddHyper = self.dEmbeds @ self.dHyper
            ggHyper = self.gEmbeds @ self.gHyper

        for i in range(args.gnn_layer):
            gatEmbeds = self.gatLite(self.edgeDropper(adj, keepRate), embedsLst[-1])
            hyperDEmbeds = self.gatedGCN(ddHyper, embedsLst[-1][:args.drug])
            hyperGEmbeds = self.gatedGCN(ggHyper, embedsLst[-1][args.drug:])
            hyperEmbeds = t.concat([hyperDEmbeds, hyperGEmbeds], axis=0)

            gate_value = self.gate(gatEmbeds)
            fusedEmbeds = gate_value * gatEmbeds + (1 - gate_value) * hyperEmbeds

            gatEmbedsLst.append(gatEmbeds)
            hyperEmbedsLst.append(hyperEmbeds)
            embedsLst.append(fusedEmbeds)

        embeds = sum(embedsLst)
        return embeds, gatEmbedsLst, hyperEmbedsLst

    def calcLosses(self, drugs, genes, labels, adj, keepRate):
        embeds, gtEmbedsLst, mpHGNNEmbedsLst = self.forward(adj, keepRate)
        dEmbeds, gEmbeds = embeds[:args.drug], embeds[args.drug:]
        dEmbeds = dEmbeds[drugs]
        gEmbeds = gEmbeds[genes]

        pre = self.classifierLayer(dEmbeds, gEmbeds)
        ceLoss = ce(pre, labels)

        sslLoss = 0
        for i in range(1, args.gnn_layer + 1):
            embeds1 = gtEmbedsLst[i].detach()
            embeds2 = mpHGNNEmbedsLst[i]
            sslLoss += contrastLoss(embeds1[:args.drug], embeds2[:args.drug], t.unique(drugs), args.temp)
            sslLoss += contrastLoss(embeds1[args.drug:], embeds2[args.drug:], t.unique(genes), args.temp)

        return ceLoss, sslLoss

    def calcLosses2(self, drugs, genes, labels, adj, keepRate):
        embeds, gtEmbedsLst, mpHGNNEmbedsLst = self.forward(adj, keepRate)
        dEmbeds, gEmbeds = embeds[:args.drug], embeds[args.drug:]
        dEmbeds = dEmbeds[drugs]
        gEmbeds = gEmbeds[genes]

        pre = self.classifierLayer(dEmbeds, gEmbeds)
        ceLoss = ce(pre, labels)

        sslLoss = 0
        for i in range(1, args.gnn_layer + 1):
            e1 = gtEmbedsLst[i]
            e2 = mpHGNNEmbedsLst[i]
            sslLoss += contrastLoss(e1[:args.drug], e2[:args.drug], t.unique(drugs), args.temp)
            sslLoss += contrastLoss(e2[:args.drug], e1[:args.drug], t.unique(drugs), args.temp)
            sslLoss += contrastLoss(e1[args.drug:], e2[args.drug:], t.unique(genes), args.temp)
            sslLoss += contrastLoss(e2[args.drug:], e1[args.drug:], t.unique(genes), args.temp)
        sslLoss = sslLoss / (4 * args.gnn_layer)
        return ceLoss, sslLoss

    def predict(self, adj, drugs, genes):
        embeds, _, _ = self.forward(adj, 1.0)
        dEmbeds, gEmbeds = embeds[:args.drug], embeds[args.drug:]
        dEmbeds = dEmbeds[drugs]
        gEmbeds = gEmbeds[genes]
        pre = self.classifierLayer(dEmbeds, gEmbeds)
        return pre


class EdgeDropout(nn.Module):
    def __init__(self):
        super(EdgeDropout, self).__init__()

    def forward(self, adj, keepRate):
        if keepRate == 1.0:
            return adj
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        mask = ((t.rand(edgeNum) + keepRate).floor()).type(t.bool)
        newVals = vals[mask] / keepRate
        newIdxs = idxs[:, mask]
        return t.sparse.FloatTensor(newIdxs, newVals, adj.shape)


class ClassifierLayer(nn.Module):
    def __init__(self, initializer='glorot_uniform'):
        super(ClassifierLayer, self).__init__()
        self.lin1 = nn.Linear(args.latdim * 2, 128)
        self.lin2 = nn.Linear(128, args.num_classes)

        # Apply Keras-style initialization to the weights
        keras_style_init(self.lin1.weight, initializer)
        keras_style_init(self.lin1.bias, 'zeros')  # Biases are typically initialized to zeros in Keras
        keras_style_init(self.lin2.weight, initializer)
        keras_style_init(self.lin2.bias, 'zeros')

    def forward(self, dEmbeds, gEmbeds):
        embeds = t.concat((dEmbeds, gEmbeds), 1)
        embeds = F.relu(self.lin1(embeds))
        embeds = F.dropout(embeds, p=0.4, training=self.training)
        ret = self.lin2(embeds)
        return ret


class SpAdjDropEdge(nn.Module):
    def __init__(self):
        super(SpAdjDropEdge, self).__init__()

    def forward(self, adj, keepRate):
        if keepRate == 1.0:
            return adj
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        mask = ((t.rand(edgeNum) + keepRate).floor()).type(t.bool)
        newVals = vals[mask] / keepRate
        newIdxs = idxs[:, mask]
        return t.sparse.FloatTensor(newIdxs, newVals, adj.shape)


class GATLite(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.6, alpha=0.2):
        """
        Lightweight graph attention layer over a sparse adjacency matrix.
        """
        super(GATLite, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_vector = nn.Parameter(t.Tensor(2 * out_dim, 1))
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attn_vector)

    def forward(self, adj, embeds):
        """
        Args:
            adj: sparse adjacency matrix with shape (N, N)
            embeds: node embeddings with shape (N, in_dim)
        """
        h = self.linear(embeds)
        N = h.size(0)

        indices = adj._indices()

        h_i = h[indices[0]]
        h_j = h[indices[1]]
        edge_features = t.cat([h_i, h_j], dim=1)
        e = self.leakyrelu(t.matmul(edge_features, self.attn_vector)).squeeze()

        exp_e = t.exp(e)
        denom = t.zeros((N,), device=embeds.device)
        denom = denom.index_add(0, indices[0], exp_e)
        alpha = exp_e / (denom[indices[0]] + 1e-16)
        alpha = self.dropout(alpha)

        h_prime = t.zeros_like(h)
        h_prime = h_prime.index_add(0, indices[0], h[indices[1]] * alpha.unsqueeze(-1))
        h_prime = F.elu(h_prime)
        return l2_norm(h_prime)


class GatedGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=2, dropout=0.4):
        """
        Gated graph convolution block for incidence-style hypergraph inputs.
        """
        super(GatedGCN, self).__init__()
        layers = []
        input_dim = in_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, in_dim))
        self.mlp = nn.Sequential(*layers)

        self.gate_linear = nn.Linear(in_dim * 2, in_dim)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.gate_linear.weight)
        nn.init.zeros_(self.gate_linear.bias)

    def forward(self, H, embeds):
        """
        Args:
            H: sparse incidence matrix with shape (N, num_hyperedges)
            embeds: node embeddings with shape (N, in_dim)
        """
        H = H.to(embeds.device)
        node_transformed = self.mlp(embeds)

        hyperedge_features = t.spmm(H.T, node_transformed)
        hyperedge_features = self.mlp(hyperedge_features)

        high_order_feats = t.spmm(H, hyperedge_features)
        high_order_feats = F.elu(high_order_feats)

        gate_input = t.cat([embeds, high_order_feats], dim=1)
        gate = t.sigmoid(self.gate_linear(gate_input))

        out = gate * high_order_feats + (1 - gate) * embeds
        return l2_norm(out)


class EnhancedClassifierLayer(t.nn.Module):
    def __init__(self, initializer='glorot_uniform'):
        super(EnhancedClassifierLayer, self).__init__()
        self.lin1 = t.nn.Linear(args.latdim * 2, 256)
        self.lin2 = t.nn.Linear(256, 128)
        self.lin3 = t.nn.Linear(128, args.num_classes)
        self.dropout = t.nn.Dropout(0.5)
        self.bn1 = t.nn.BatchNorm1d(256)
        self.bn2 = t.nn.BatchNorm1d(128)

        # Apply Keras-style initialization to weights
        keras_style_init(self.lin1.weight, initializer)
        keras_style_init(self.lin1.bias, 'zeros')
        keras_style_init(self.lin2.weight, initializer)
        keras_style_init(self.lin2.bias, 'zeros')
        keras_style_init(self.lin3.weight, initializer)
        keras_style_init(self.lin3.bias, 'zeros')

    def forward(self, dEmbeds, gEmbeds):
        embeds = t.concat((dEmbeds, gEmbeds), 1)
        embeds = F.relu(self.bn1(self.lin1(embeds)))
        embeds = self.dropout(embeds)
        embeds = F.relu(self.bn2(self.lin2(embeds)))
        embeds = self.dropout(embeds)
        ret = self.lin3(embeds)
        return ret
