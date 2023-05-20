from transformers import BertModel, BertTokenizer, BertConfig
import torch
from torch import nn

class Bert4CLS(nn.Module):
    def __init__(self, Bert):
        super().__init__()
        self.Backbone = Bert

    def freeze(self):
        for name, para in self.Backbone.named_parameters():
            para.requires_grad_(False)

    def unfreeze(self):
        for name, para in self.Backbone.named_parameters():
            para.requires_grad_(True)

    def forward(self, seq, mask):
        x = self.Backbone(input_ids=seq, attention_mask=mask)[0]

        return x

