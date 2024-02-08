# OutfitGAN: compatibility network
# pretrained CNN for feature extracting
# Here we choose a pretrained InceptionV3 that finetuned on iFashion dataset

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_, xavier_uniform_
import torch.nn.functional as F
from torchvision.models import inception_v3
from torch.utils.model_zoo import load_url
from itertools import combinations, permutations
import numpy as np

class FashionEvaluator(nn.Module):
    def __init__(self, cnn_feat_dim):
        super(FashionEvaluator, self).__init__()

        self.feat_layer = nn.Linear(cnn_feat_dim, 1024)
        self.emb_layer = nn.Sequential(
            nn.Linear(2048, 512),  # layer1
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(512, 512),  # layer2
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(512, 256),  # layer3
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(256, 256),  # layer4
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.35)
        )
        self.eval_layer = nn.Sequential(
            nn.Linear(256, 128),  # layer1
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(128, 128),  # layer2
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(128, 32),  # layer3
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(32, 1)  # softmax layer
            # nn.Sigmoid()  # regularize the output to [0,1]
        )
        self.apply(xavier_normal_initialization)
    
    def outfit_emb(self, cnn_feats):  
        # cnn_feats: [o_num, 4, 2048]
        combs = combinations(np.arange(cnn_feats.shape[1]), 2)
        # combs = permutations(np.arange(cnn_feats.shape[1]), 2)
        combs = [list(comb) for comb in combs]

        o_embs = []
        for o_feats in cnn_feats:  # [4, 2048]
            feats = self.feat_layer(o_feats)  # [4, feat_dim]
            # feats = o_feats.clone()
            comb_feats = torch.stack([feats[comb].reshape(-1) for comb in combs])  # [6, feat_dim * 2]
            relation_embs = self.emb_layer(comb_feats)  # [6, emb_dim]

            o_emb = torch.mean(relation_embs, dim=0)  # [emb_dim]
            o_embs.append(o_emb)

        return torch.stack(o_embs)  # [o_num, emb_dim]
    
    def pred_score(self, o_embs):
        predictions = self.eval_layer(o_embs)
        
        return predictions.view(-1)

    def forward(self, cnn_feats):
        o_embs = self.outfit_emb(cnn_feats)
        predictions = self.eval_layer(o_embs)

        return predictions.view(-1)  # [o_num, 1]

class InceptionV3(nn.Module):
    def __init__(self, model_path, num_classes):
        super(InceptionV3, self).__init__()
        self.model = inception_v3()
        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        self.model.load_state_dict(load_url(url, map_location=lambda storage, loc: storage))

        self.num_classes = num_classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        num_features_aux = self.model.AuxLogits.fc.in_features
        self.model.AuxLogits.fc = nn.Linear(num_features_aux, num_classes)
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()
    
    def forward(self, x):
        x = self.model(x)
        x = nn.Softmax(dim=1)(x)
        return x
    
    @torch.no_grad()
    def feature_extract(self, x):  # N x 3 x 299 x 299 # TODO: 记得输入resize并且在normalize
        x = self.model._transform_input(x)
        features = self.extractor(x)  # N x 2048
        return features
    
    def extractor(self, x):
        # N x 3 x 299 x 299
        x = self.model.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.model.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.model.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.model.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.model.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.model.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.model.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.model.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.model.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.model.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.model.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.model.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.model.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.model.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.model.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.model.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.model.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.model.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.model.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.model.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        return x

def xavier_normal_initialization(module):
    r""" using `xavier_normal_`_ in PyTorch to initialize the parameters in
    nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
    using constant 0 to initialize.
    .. _`xavier_normal_`:
        https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_normal_#torch.nn.init.xavier_normal_
    Examples:
        >>> self.apply(xavier_normal_initialization)
    """
    if isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)

    elif isinstance(module, nn.Embedding):
        xavier_normal_(module.weight.data)

    elif isinstance(module, nn.Sequential):
        for mod in module:
            if isinstance(mod, nn.Linear):
                xavier_normal_(mod.weight.data)
                if mod.bias is not None:
                    constant_(mod.bias.data, 0)
                    
            elif isinstance(mod, nn.Embedding):
                xavier_normal_(mod.weight.data)

def xavier_uniform_initialization(module):
    r""" using `xavier_normal_`_ in PyTorch to initialize the parameters in
    nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
    using constant 0 to initialize.
    .. _`xavier_normal_`:
        https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_normal_#torch.nn.init.xavier_normal_
    Examples:
        >>> self.apply(xavier_normal_initialization)
    """
    if isinstance(module, nn.Linear):
        xavier_uniform_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)

    elif isinstance(module, nn.Embedding):
        xavier_uniform_(module.weight.data)

    elif isinstance(module, nn.Sequential):
        for mod in module:
            if isinstance(mod, nn.Linear):
                xavier_uniform_(mod.weight.data)
                if mod.bias is not None:
                    constant_(mod.bias.data, 0)
                    
            elif isinstance(mod, nn.Embedding):
                xavier_uniform_(mod.weight.data)
    
