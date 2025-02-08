import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import EvolveGCNO, EvolveGCNH, GCLSTM, LRGCN
from torch_geometric.utils import get_laplacian, to_dense_adj, dense_to_sparse
from torch_geometric.nn import GCNConv
import copy

class EvolveGCNModel(nn.Module):
    """
    EvolveGCN model
    """
    def __init__(self, dim_node_features, dim_edge_features, dim_target, readout_class, config):
        """
        Initializes the model.
        :param dim_node_features: input node features
        :param dim_edge_features: input edge features
        :param dim_target: target size
        :param readout_class: the class of the predictor that will classify node/graph embeddings produced by>
        :param config: the configuration dictionary to extract further hyper-parameters
        """

        super().__init__()

        self.dim_node_features = dim_node_features
        self.dim_target = dim_target
        #self.version = config['encoder_version']
        
        self.predictor = readout_class(dim_node_features = self.dim_node_features,
                                         dim_edge_features = dim_edge_features,
                                         dim_target = dim_target,
                                         config = config)

    def forward(self, snapshot, prev_state=None):
        x, edge_index, mask = snapshot.x, snapshot.edge_index, snapshot.mask
        
        node_mask = snapshot.node_mask if hasattr(snapshot, 'node_mask') else None

        h = self.model(x, edge_index)
        h = torch.relu(h)

        if 'link_pred_ids' in snapshot:
            source, target = snapshot.link_pred_ids
            h_cat = torch.cat((h[source], h[target]),
                                     dim=-1)
        else:
            h_cat = h
        out, _ = self.predictor(h_cat, None)

        if node_mask is not None:
            out = out[node_mask]

        #if mask.shape[0] > 1:
        #    out = out[mask]

        return out, h


class EvolveGCN_H_Model(EvolveGCNModel):
    def __init__(self, dim_node_features, dim_edge_features, dim_target, readout_class, config):
        super().__init__(dim_node_features, dim_edge_features, dim_target, readout_class, config)
        num_nodes = config['num_nodes']
        normalize = config['normalize']
        self.model = EvolveGCNH(num_of_nodes = num_nodes,
                                in_channels = self.dim_node_features,
                                normalize = normalize)


class EvolveGCN_O_Model(EvolveGCNModel):
    def __init__(self, dim_node_features, dim_edge_features, dim_target, readout_class, config):
        super().__init__(dim_node_features, dim_edge_features, dim_target, readout_class, config)
        normalize = config['normalize']
        self.model = EvolveGCNO(in_channels = self.dim_node_features,
                                normalize = normalize)


class GCLSTMModel(nn.Module):
    """
    GCLSTM model
    """
    def __init__(self, dim_node_features, dim_edge_features, dim_target, readout_class, config):
        """
        Initializes the model.
        :param dim_node_features: input node features
        :param dim_edge_features: input edge features
        :param dim_target: target size
        :param readout_class: the class of the predictor that will classify node/graph embeddings produced by>
        :param config: the configuration dictionary to extract further hyper-parameters
        """

        super().__init__()

        self.dim_node_features = dim_node_features
        self.dim_target = dim_target

        self.dim_embedding = config['dim_embedding']
        self.k = config['K']
        self.normalization = config.get('normalization', None)

        self.model = GCLSTM(in_channels = self.dim_node_features,
                            out_channels = self.dim_embedding,
                            K = self.k,
                            normalization = self.normalization)

        self.predictor = readout_class(dim_node_features = self.dim_embedding,
                                         dim_edge_features = dim_edge_features,
                                         dim_target = dim_target,
                                         config = config)

    def forward(self, snapshot, prev_state=(None,None)):
        x, edge_index, mask = snapshot.x, snapshot.edge_index, snapshot.mask
        node_mask = snapshot.node_mask if hasattr(snapshot, 'node_mask') else None

        h, c = prev_state if prev_state is not None else (None, None)

        _, edge_weight = get_laplacian(edge_index, normalization=self.normalization)
        h, c = self.model(x, edge_index, H=h, C=c, lambda_max=edge_weight.max())
        h = torch.relu(h)

        if 'link_pred_ids' in snapshot:
            source, target = snapshot.link_pred_ids
            h_cat = torch.cat((h[source], h[target]),
                                     dim=-1)
        else:
            h_cat = h
        out, _ = self.predictor(h_cat, None)

        if node_mask is not None:
            out = out[node_mask]

        #if mask.shape[0] > 1:
        #    out = out[mask]

        return out, torch.stack((h, c), dim=0)


class LRGCNModel(nn.Module):
    """
    LRGCN model
    """
    def __init__(self, dim_node_features, dim_edge_features, dim_target, readout_class, config):
        """
        Initializes the model.
        :param dim_node_features: input node features
        :param dim_edge_features: input edge features
        :param dim_target: target size
        :param readout_class: the class of the predictor that will classify node/graph embeddings produced by>
        :param config: the configuration dictionary to extract further hyper-parameters
        """

        super().__init__()

        self.dim_node_features = dim_node_features
        self.dim_target = dim_target

        self.dim_embedding = config['dim_embedding']
        self.model = LRGCN(in_channels = self.dim_node_features,
                           out_channels = self.dim_embedding,
                           num_relations = config['num_relations'],
                           num_bases = config['num_bases'])

        self.predictor = readout_class(dim_node_features = self.dim_embedding,
                                         dim_edge_features = dim_edge_features,
                                         dim_target = dim_target,
                                         config = config)

    def forward(self, snapshot, prev_state=None):
        x, edge_index, edge_type, mask = snapshot.x, snapshot.edge_index, snapshot.relation_type, snapshot.mask
        node_mask = snapshot.node_mask if hasattr(snapshot, 'node_mask') else None

        h, c = prev_state if prev_state is not None else (None, None)

        h, c = self.model(x, edge_index, edge_type, H=h, C=c)
        h = torch.relu(h)

        if 'link_pred_ids' in snapshot:
            source, target = snapshot.link_pred_ids
            h_cat = torch.cat((h[source], h[target]),
                                     dim=-1)
        else:
            h_cat = h
        out, _ = self.predictor(h_cat, None)

        if node_mask is not None:
            out = out[node_mask]

        #if mask.shape[0] > 1:
        #    out = out[mask]

        return out, torch.stack((h, c), dim=0)
    
# Contribution for the DL project

class RolandGNNModel(nn.Module):
    """
    ROLAND-inspired GNN model for dynamic graphs.
    """
    def __init__(self, dim_node_features, dim_edge_features, dim_target, readout_class, config):
        """
        Initializes the model.
        :param dim_node_features: Dimension of input node features
        :param dim_edge_features: Dimension of input edge features
        :param dim_target: Dimension of the target output
        :param readout_class: Class for the readout/predictor that will classify node/graph embeddings
        :param config: Configuration dictionary for hyperparameters
        """
        super(RolandGNNModel, self).__init__()

        self.dim_node_features = dim_node_features
        self.dim_edge_features = dim_edge_features
        self.dim_target = dim_target
        self.hidden_dim = config.get('hidden_dim', 64)
        self.num_layers = config.get('num_layers', 2)

        # Define GNN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(dim_node_features, self.hidden_dim))
        for _ in range(self.num_layers - 1):
            self.convs.append(GCNConv(self.hidden_dim, self.hidden_dim))

        # Embedding update module
        self.embedding_updater = nn.GRU(self.hidden_dim, self.hidden_dim)

        # Predictor
        self.predictor = readout_class(
            dim_node_features=self.hidden_dim,
            dim_edge_features=dim_edge_features,
            dim_target=dim_target,
            config=config
        )

    def forward(self, snapshot, prev_state=None):
        """
        Forward pass through the model.
        :param snapshot: A data object representing the current graph snapshot
        :param prev_state: Previous hidden state for the embedding updater (if any)
        :return: Output predictions and updated hidden state
        """
        x, edge_index = snapshot.x, snapshot.edge_index

        # Apply GNN layers
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))

        # Update node embeddings over time
        if prev_state is None:
            prev_state = torch.zeros(1, x.size(0), self.hidden_dim, device=x.device)
        x, hidden = self.embedding_updater(x.unsqueeze(0), prev_state)

        # Readout/prediction
        if 'link_pred_ids' in snapshot:
            source, target = snapshot.link_pred_ids
            h_cat = torch.cat((x[0][source], x[0][target]), dim=-1)
        else:
            h_cat = x[0]
        out, _ = self.predictor(h_cat, None)

        return out, hidden
    

class TimeAwareRandomWalkDiffusion(nn.Module):
    """
    Time-aware Random Walk Diffusion for dynamic graphs.
    """
    def __init__(self, alpha=0.1):
        """
        Initializes the diffusion model.
        :param alpha: Temporal decay factor
        """
        super(TimeAwareRandomWalkDiffusion, self).__init__()
        self.alpha = alpha

    def forward(self, edge_index, edge_time, num_nodes):
        """
        Applies time-aware random walk diffusion to the adjacency matrix.
        :param edge_index: Edge indices
        :param edge_time: Timestamps associated with each edge
        :param num_nodes: Number of nodes in the graph
        :return: Diffused adjacency matrix in sparse format
        """
        # Convert edge_index to dense adjacency matrix
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]

        # Apply temporal decay to each edge based on its timestamp
        current_time = edge_time.max()
        decay = torch.exp(-self.alpha * (current_time - edge_time))
        for i, (u, v) in enumerate(edge_index.t()):
            adj[u, v] *= decay[i]

        # Normalize adjacency matrix
        deg = adj.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)

        # Convert back to sparse format
        edge_index, edge_weight = dense_to_sparse(adj)

        return edge_index, edge_weight
    
class TiaRaGNNModel(nn.Module):
    """
    GNN model with Time-aware Random Walk Diffusion for dynamic graphs.
    """
    def __init__(self, dim_node_features, dim_edge_features, dim_target, readout_class, config):
        """
        Initializes the model.
        :param dim_node_features: Dimension of input node features
        :param dim_edge_features: Dimension of input edge features
        :param dim_target: Dimension of the target output
        :param readout_class: Class for the readout/predictor that will classify node/graph embeddings
        :param config: Configuration dictionary for hyperparameters
        """
        super(TiaRaGNNModel, self).__init__()

        self.dim_node_features = dim_node_features
        self.dim_edge_features = dim_edge_features
        self.dim_target = dim_target
        self.hidden_dim = config.get('hidden_dim', 64)
        self.num_layers = config.get('num_layers', 2)
        self.alpha = config.get('alpha', 0.1)

        # Time-aware Random Walk Diffusion
        self.diffusion = TimeAwareRandomWalkDiffusion(alpha=self.alpha)

        # Define GNN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(dim_node_features, self.hidden_dim))
        for _ in range(self.num_layers - 1):
            self.convs.append(GCNConv(self.hidden_dim, self.hidden_dim))

        # Predictor
        self.predictor = readout_class(
            dim_node_features=self.hidden_dim,
            dim_edge_features=dim_edge_features,
            dim_target=dim_target,
            config=config
        )

    def forward(self, snapshot):
        """
        Forward pass through the model.
        :param snapshot: A data object representing the current graph snapshot
        :return: Output predictions
        """
        x, edge_index, edge_time = snapshot.x, snapshot.edge_index, snapshot.edge_time
        num_nodes = x.size(0)

        # Apply time-aware random walk diffusion
        edge_index, edge_weight = self.diffusion(edge_index, edge_time, num_nodes)

        # Apply GNN layers
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight))

        # Readout/prediction
        out, _ = self.predictor(x, None)

        return out
    
class DynamicGEMModel(nn.Module):
    """
    DynamicGEM-inspired GNN model for dynamic graphs.
    """
    def __init__(self, dim_node_features, dim_edge_features, dim_target, readout_class, config):
        """
        Initializes the model.
        :param dim_node_features: Dimension of input node features
        :param dim_edge_features: Dimension of input edge features
        :param dim_target: Dimension of the target output
        :param readout_class: Class for the readout/predictor that will classify node/graph embeddings
        :param config: Configuration dictionary for hyperparameters
        """
        super(DynamicGEMModel, self).__init__()

        self.dim_node_features = dim_node_features
        self.dim_edge_features = dim_edge_features
        self.dim_target = dim_target
        self.hidden_dim = config.get('hidden_dim', 64)
        self.num_layers = config.get('num_layers', 2)

        # Define GNN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(dim_node_features, self.hidden_dim))
        for _ in range(self.num_layers - 1):
            self.convs.append(GCNConv(self.hidden_dim, self.hidden_dim))

        # Embedding update module
        self.embedding_updater = nn.GRU(self.hidden_dim, self.hidden_dim)

        # Predictor
        self.predictor = readout_class(
            dim_node_features=self.hidden_dim,
            dim_edge_features=dim_edge_features,
            dim_target=dim_target,
            config=config
        )

    def forward(self, snapshot, prev_state=None):
        """
        Forward pass through the model.
        :param snapshot: A data object representing the current graph snapshot
        :param prev_state: Previous hidden state for the embedding updater (if any)
        :return: Output predictions and updated hidden state
        """
        x, edge_index = snapshot.x, snapshot.edge_index

        # Apply GNN layers
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))

        # Update node embeddings over time
        if prev_state is None:
            prev_state = torch.zeros(1, x.size(0), self.hidden_dim, device=x.device)
        x, hidden = self.embedding_updater(x.unsqueeze(0), prev_state)

        # Readout/prediction
        out, _ = self.predictor(x[0], None)

        return out, hidden