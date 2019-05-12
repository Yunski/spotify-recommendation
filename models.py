import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Variable

from layers import conv_layer, fc_layer, glorot_uniform, lecunn_uniform

class MF(nn.Module):
    def __init__(self, n_users, n_items, n_features, n_track_features, with_bias=False, with_audio_feats=False):
        super(MF, self).__init__()
        self.name = "mf"
        self.n_users = n_users
        self.n_items = n_items
        self.n_features = n_features
        self.n_track_features = n_track_features
        self.with_bias = with_bias
        self.user_embeddings = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.n_features)
        self.item_embeddings = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.n_features)
        self.user_embeddings.weight.data.normal_(0, 0.01)
        self.item_embeddings.weight.data.normal_(0, 0.01)
        if with_bias:
            self.user_biases = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=1)
            self.item_biases = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=1)
    
    def forward(self, user_indices, item_indices, track_features=None):
        user_embeddings = self.user_embeddings(user_indices)
        item_embeddings = self.item_embeddings(item_indices) 
        pred = torch.mul(user_embeddings, item_embeddings).sum(1).unsqueeze(1)
        if self.with_bias:
            user_bias = self.user_biases(user_indices)
            item_bias = self.item_biases(item_indices)
            pred += user_bias + item_bias
        return pred

class GMF(nn.Module):
    def __init__(self, n_users, n_items, n_features, n_track_features, with_bias=False, with_audio_feats=False):
        super(GMF, self).__init__()
        self.name = "gmf"
        self.n_users = n_users
        self.n_items = n_items
        self.n_features = n_features
        self.n_track_features = n_track_features
        self.with_bias = with_bias
        self.user_embeddings = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.n_features)
        self.item_embeddings = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.n_features)
        self.user_embeddings.weight.data.normal_(0, 0.01)
        self.item_embeddings.weight.data.normal_(0, 0.01)
        self.linear = nn.Linear(self.n_features, 1)
        if with_bias:
            self.user_biases = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=1)
            self.item_biases = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=1)
            self.user_biases.weight.data.normal_(0, 0.01)
            self.item_biases.weight.data.normal_(0, 0.01)
    
    def forward(self, user_indices, item_indices, track_features=None):
        user_embeddings = self.user_embeddings(user_indices)
        item_embeddings = self.item_embeddings(item_indices) 
        logits = torch.mul(user_embeddings, item_embeddings)
        if self.with_bias:
            user_bias = self.user_biases(user_indices)
            item_bias = self.item_biases(item_indices)
            logits += user_bias + item_bias
        return self.linear(logits)

class MLP(nn.Module):
    def __init__(self, n_users, n_items, n_features, n_track_features, 
        mlp_layer=fc_layer, with_bias=False, with_audio_feats=False):
        super(MLP, self).__init__()
        self.name = "mlp"
        self.n_users = n_users
        self.n_items = n_items
        self.n_features = n_features
        self.n_track_features = n_track_features
        self.with_bias = with_bias
        self.user_embeddings = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.n_features)
        self.item_embeddings = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.n_features)
        self.user_embeddings.weight.data.normal_(0, 0.01)
        self.item_embeddings.weight.data.normal_(0, 0.01)
        layers = [32 // (2**i) for i in range(6) if 32 // (2**i) >= self.n_features]
        mlp_layers = [mlp_layer(2 * self.n_features, 32, dropout=True)]
        for inp_dim, out_dim in zip(layers, layers[1:]):
            mlp_layers.append(mlp_layer(inp_dim, out_dim, dropout=True))
        self.layers = nn.Sequential(*mlp_layers)
        self.linear = mlp_layer(self.n_features, 1, linear=True)
        if with_bias:
            self.user_biases = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=1)
            self.item_biases = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=1)
            self.user_biases.weight.data.normal_(0, 0.01)
            self.item_biases.weight.data.normal_(0, 0.01)
    
    def forward(self, user_indices, item_indices, track_features=None):
        user_embeddings = self.user_embeddings(user_indices)
        item_embeddings = self.item_embeddings(item_indices)
        cat = torch.cat([user_embeddings, item_embeddings], dim=1)
        logits = self.layers(cat)
        if self.with_bias:
            user_bias = self.user_biases(user_indices)
            item_bias = self.item_biases(item_indices)
            logits += user_bias + item_bias
        return self.linear(logits)

class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, n_features, n_track_features, 
        mlp_layer=fc_layer, with_bias=False, with_audio_feats=False):
        super(NeuMF, self).__init__()
        self.name = "neumf"
        self.n_users = n_users
        self.n_items = n_items
        self.n_features = n_features
        self.n_track_features = n_track_features
        self.with_bias = with_bias
        self.with_audio_feats = with_audio_feats 
        self.user_embeddings_gmf = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.n_features)
        self.item_embeddings_gmf = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.n_features)
        self.user_embeddings_mlp = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.n_features)
        self.item_embeddings_mlp = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.n_features)
        self.user_embeddings_gmf.weight.data.normal_(0, 0.01)
        self.item_embeddings_gmf.weight.data.normal_(0, 0.01)
        self.user_embeddings_mlp.weight.data.normal_(0, 0.01)
        self.item_embeddings_mlp.weight.data.normal_(0, 0.01)

        layers = [32 // (2**i) for i in range(6) if 32 // (2**i) >= n_features]
        mlp_layers = [mlp_layer(2 * self.n_features, 32, dropout=True)]
        for inp_dim, out_dim in zip(layers, layers[1:]):
            mlp_layers.append(mlp_layer(inp_dim, out_dim, dropout=True))
        self.mlp = nn.Sequential(*mlp_layers)
        self.linear = mlp_layer(2 * self.n_features, 1, linear=True)

        if with_bias:
            self.user_biases = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=1)
            self.item_biases = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=1)
            self.user_biases.weight.data.normal_(0, 0.01)
            self.item_biases.weight.data.normal_(0, 0.01)
        elif with_audio_feats:
            self.user_biases = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=1)
            self.user_biases.weight.data.normal_(0, 0.01)
            self.item_biases = nn.Sequential(
                mlp_layer(self.n_track_features, self.n_track_features, dropout=True),
                mlp_layer(self.n_track_features, 8, dropout=True),
                mlp_layer(8, 1, linear=True)
            )

    def forward(self, user_indices, item_indices, track_features=None):
        user_embeddings_mlp = self.user_embeddings_mlp(user_indices)
        item_embeddings_mlp = self.item_embeddings_mlp(item_indices)
        user_embeddings_gmf = self.user_embeddings_gmf(user_indices)
        item_embeddings_gmf = self.item_embeddings_gmf(item_indices)
        mlp_features = torch.cat([user_embeddings_mlp, item_embeddings_mlp], dim=1)
        mlp_features = self.mlp(mlp_features)
        gmf_features = torch.mul(user_embeddings_gmf, item_embeddings_gmf)
        features = torch.cat([mlp_features, gmf_features], dim=1)
        logits = self.linear(features)
        if self.with_bias or self.with_audio_feats:
            user_bias = self.user_biases(user_indices)
            item_bias = self.item_biases(item_indices)
            logits += user_bias + item_bias
        return logits

class ConvNCF(nn.Module):
    def __init__(self, n_users, n_items, n_features, n_track_features, 
        layers=None, mlp_layer=fc_layer, with_bias=False, with_audio_feats=False):
        super(ConvNCF, self).__init__()
        self.name = "convncf"
        self.n_users = n_users
        self.n_items = n_items
        self.n_features = n_features
        self.n_track_features = n_track_features
        self.with_bias = with_bias
        self.user_embeddings= torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.n_features)
        self.item_embeddings = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.n_features)
        n_layers = int(np.log2(self.n_features))
        if layers is None:
            self.layers = [32] * n_layers
        else:
            self.layers = layers
        conv_layers = []
        for inp_dim, out_dim in zip([1] + self.layers[:-1], self.layers):
            conv_layers.append(conv_layer(inp_dim, out_dim, k=2, stride=2))
        self.cnn = nn.Sequential(*conv_layers)
        self.linear = nn.Linear(self.layers[-1], 1)
        if with_bias:
            self.user_biases = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=1)
            self.item_biases = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=1)
            self.user_biases.weight.data.normal_(0, 0.01)
            self.item_biases.weight.data.normal_(0, 0.01)
 
    def forward(self, user_indices, item_indices, track_features):
        user_embeddings = self.user_embeddings(user_indices)
        item_embeddings = self.item_embeddings(item_indices)
        outer_prod = user_embeddings.unsqueeze(2) * item_embeddings.unsqueeze(1)
        outer_prod = outer_prod.unsqueeze(1)
        features = self.cnn(outer_prod)
        features = features.view(features.shape[0], -1)
        logits = self.linear(features)
        if self.with_bias:
            user_bias = self.user_biases(user_indices)
            item_bias = self.item_biases(item_indices)
            logits += user_bias + item_bias
        return logits

