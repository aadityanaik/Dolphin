# ------------------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. All Right reserved.
# ------------------------------------------------------------------------------
import torch
from torch import nn
from transformers import DistilBertModel, DistilBertConfig
import torch.nn.functional as F
from torchvision.transforms.functional import normalize, resize
from einops import rearrange
from transformers import DistilBertTokenizer
import numpy as np
from scipy import signal
import os
from s3d import S3D

class Projection(nn.Module):
    def __init__(self, d_in, d_out=256, p=0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds

class VideoEncoder(nn.Module):
    """
    Encode videos to a fixed size vector
    """

    def __init__(self, pretrained, trainable):
        super().__init__()

        self.model = S3D(400)
        self.embedding_dim = list(self.model.fc.children())[0].in_channels
        self.model.fc = nn.Identity()
        for p in self.model.parameters():
            p.requires_grad = trainable

    def preprocess(self, x):
        B, T, H, W, C = x.shape
        if T != 2:
            x = F.interpolate(rearrange(x, "b t h w c -> b c t h w"), size=[2, H, W])
            x = rearrange(x, "b c t h w -> b t h w c")
        assert C == 3
        x = rearrange(x, "b t h w c -> (b t) c h w")
        x = resize(x, (224, 224)) if H != 224 and W != 224 else x
        # this is a rgb video, just normalize
        x = x.float() / 255.
        # convert to BCTHW
        x = normalize(x, mean=(0.43216, 0.394666, 0.37645), std=(0.22803, 0.22145, 0.216989))
        x = rearrange(x, "(b t) c h w -> b c t h w", b = B)
        return x

    def forward(self, x):
        x = self.preprocess(x)
        return self.model(x)

class TextEncoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", pretrained=True, trainable=True, max_length=200):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())

        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, raw_text):
        batch_encoding = self.tokenizer(raw_text, padding=True, truncation=True, max_length=self.max_length)
        input_ids = torch.tensor(batch_encoding['input_ids']).cuda()
        attention_mask = torch.tensor(batch_encoding['attention_mask']).cuda()
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=256,
        dropout=0.1
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class MLPClassifier(nn.Module):
  def __init__(self, input_dim, latent_dim, output_dim, n_layers, dropout_rate):
    super(MLPClassifier, self).__init__()

    layers = []
    layers.append(nn.Linear(input_dim, latent_dim))
    layers.append(nn.ReLU())
    layers.append(nn.BatchNorm1d(latent_dim))
    layers.append(nn.Dropout(dropout_rate))
    for _ in range(n_layers - 1):
      layers.append(nn.Linear(latent_dim, latent_dim))
      layers.append(nn.ReLU())
      layers.append(nn.BatchNorm1d(latent_dim))
      layers.append(nn.Dropout(dropout_rate))
    layers.append(nn.Linear(latent_dim, output_dim))

    self.net = nn.Sequential(*layers)

  def forward(self, x):
    logits = self.net(x)
    probs = F.softmax(logits, dim=1)
    return probs
