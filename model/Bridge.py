from torch import nn
import torch

class Bridge2(nn.Module):
    def __init__(self, vBackbone, lBackbone, layers=2):
        super().__init__()
        self.resblocks = nn.ModuleList([nn.TransformerEncoderLayer(768, 8, batch_first=True) for _ in range(layers)])
        self.head = nn.Linear(768 * 2, 3129)
        self.act = nn.Sigmoid()
        self.visual = vBackbone
        self.lingual = lBackbone
        self.isFreeze = True

    def freeze(self):
        for name, para in self.visual.named_parameters():
            para.requires_grad_(False)
        for name, para in self.lingual.named_parameters():
            para.requires_grad_(False)
        self.isFreeze = True

    def unfreeze(self):
        for name, para in self.visual.named_parameters():
            para.requires_grad_(True)
        for name, para in self.lingual.named_parameters():
            para.requires_grad_(True)
        self.isFreeze = False

    def forward(self, img, seq, pad):
        I = self.visual(img)
        T = self.lingual(seq, pad)
        M = torch.concat([I, T], dim=1)
        pad = torch.concat([torch.zeros((len(pad), 50)).to(pad.device), pad], dim=1)

        for i in range(len(self.resblocks)):
            M = self.resblocks[i](M, src_key_padding_mask=pad)

        M = torch.concat([M[:, 0], M[:, 50]], dim=1)
        M = self.head(M)
        M = self.act(M)

        return M

class Bridge(nn.Module):
    def __init__(self, vBackbone, lBackbone, layers=2):
        super().__init__()
        self.resblocks = nn.ModuleList([nn.TransformerEncoderLayer(768, 8, batch_first=True) for _ in range(layers)])
        self.head = nn.Sequential(
                nn.Linear(768 * 2, 768 * 4),
                nn.LayerNorm(768 * 4),
                nn.GELU(),
                nn.Linear(768 * 4, 3129)
        )
        #self.norm = nn.LayerNorm(3129)
        self.act = nn.Softmax(dim=-1)
        self.visual = vBackbone
        self.lingual = lBackbone
        self.isFreeze = True

    def freeze(self):
        for name, para in self.visual.named_parameters():
            para.requires_grad_(False)
        for name, para in self.lingual.named_parameters():
            para.requires_grad_(False)
        self.isFreeze = True

    def unfreeze(self):
        for name, para in self.visual.named_parameters():
            para.requires_grad_(True)
        for name, para in self.lingual.named_parameters():
            para.requires_grad_(True)
        self.isFreeze = False

    def forward(self, img, seq, pad):
        I = self.visual(img)
        T = self.lingual(seq, pad)
        M = torch.concat([I, T], dim=1)
        pad = torch.concat([torch.zeros((len(pad), 50)).to(pad.device), pad], dim=1)

        for i in range(len(self.resblocks)):
            M = self.resblocks[i](M, src_key_padding_mask=pad)

        M = torch.concat([M[:, 0], M[:, 50]], dim=1)
        M = self.head(M)
        #M = self.norm(M)
        M = self.act(M)

        return M

if __name__ == "__main__":

    from ViT4CLS import ViT4CLS, vit_b_32

    vit = ViT4CLS(vit_b_32())

    from Bert4CLS import Bert4CLS

    bert = Bert4CLS(torch.load("../bert.pth"))

    bridge = Bridge(vit, bert)

    print(bridge(torch.rand(2, 3, 224, 224), torch.LongTensor([[101, 202, 234], [101, 24325, 244]]), torch.ones((2, 3))))






'''
class Bridge4CLS(nn.Module):
    def __init__(self, kinds, layers=3):
        super().__init__()
        self.TrmBlocks = nn.ModuleList([nn.TransformerEncoderLayer
                                        (d_model=768, nhead=8, batch_first=True)
                                        for _ in range(layers)])

        self.pooler = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        self.ln_1 = nn.LayerNorm(768)

        self.l1 = nn.Linear(768, kinds)
        self.gelu = nn.GELU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, VCLSs, TCLS, pad_mask):

        for i in range(len(self.TrmBlocks)):
            VCLSs = self.TrmBlocks[i](VCLSs, src_key_padding_mask=pad_mask)

        CLS = self.pooler(TCLS, VCLSs, VCLSs, need_weights=False)[0]
        CLS = self.ln_1(CLS)
        CLS = self.l1(CLS)
        return self.softmax(CLS)

class BridgeFormer(nn.Module):
    def __init__(self, ViT, Bert, Bridge):
        super().__init__()
        self.Bridge = Bridge
        self.ViT = ViT
        self.Bert = Bert

    def forward(self, imgs, txt, padMaskB, padMask):


if __name__ == "__main__":
    ViT = ViT4CLS(vit_b_16())
    Bert = Bert4CLS(torch.load("../bert.pth"))
    Bridge = Bridge4CLS(5)
    #myBF = BridgeFormer(ViT, Bert, Bridge)
    print(Bridge(torch.rand(2, 5, 768), torch.rand(2, 1, 768), torch.LongTensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])))
    #Former = BridgeFormer(ViT, Bert, Bridge)
'''







