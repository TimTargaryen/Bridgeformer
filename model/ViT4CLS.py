from torchvision.models import VisionTransformer, vit_b_32
from torch import nn
import torch

class ViT4CLS(nn.Module):
    def __init__(self, ViT):
        super().__init__()
        self.Backbone = ViT

    def freeze(self):
        for name, para in self.Backbone.named_parameters():
            para.requires_grad_(False)
            
    def unfreeze(self):
        for name, para in self.Backbone.named_parameters():
            para.requires_grad_(True)

    def forward(self, x):
        n, c, h, w = x.shape
        p = self.Backbone.patch_size
        n_h = h // p
        n_w = w // p

        x = self.Backbone.conv_proj(x)
        x = x.reshape(n, self.Backbone.hidden_dim, n_h * n_w)
        x = x.permute(0, 2, 1)
        n = x.shape[0]

        batch_class_token = self.Backbone.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.Backbone.encoder(x)

        return x


