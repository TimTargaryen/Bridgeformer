import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from .vqa import VQA
from transformers import BertTokenizer
import os
import json
import PIL.Image as Image


from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


bertTokenizer = BertTokenizer(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocab.txt"))
transform = _transform(224)


class VQAv2(Dataset):
    def __init__(self, path, mode='train'):
        super().__init__()
        self.sub = None
        self.len = 0

        ansDict = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dict.json")))
        self.ans2cls = ansDict[0]
        self.cls2ans = ansDict[1]

        if mode == 'train':
            self.len = 443757
            self.sub = 'train2014'
        elif mode == 'val':
            self.len = 214354
            self.sub = 'val2014'

        qnn = '%s%s_%s_%s_questions.json' % ('v2_', 'OpenEnded', 'mscoco', self.sub)
        ann = '%s%s_%s_annotations.json' % ('v2_', 'mscoco', self.sub)
        self.VQA = VQA(os.path.join(path, ann), os.path.join(path, qnn))
        self.path = path

    def __len__(self):
        return self.len

    def __getitem__(self, idx):  # retru
        qInfo = self.VQA.questions['questions'][idx]

        question = qInfo['question']
        imgID = qInfo['image_id']
        questionID = qInfo['question_id']
        imgPath = os.path.join(self.path, self.sub,'COCO_' + self.sub + '_' + str(imgID).zfill(12) + '.jpg')

        label = torch.zeros(3129)
        anwsers = self.VQA.loadQA(questionID)[0]['answers']

        cnt = {}
        for anwser in anwsers:
            if anwser['answer'] not in cnt.keys():
                cnt[anwser['answer']] = 1
            else:
                cnt[anwser['answer']] += 1


        for ans in cnt.keys():
            if cnt[ans] >= 3:
                label[self.ans2cls.get(ans, self.ans2cls['<unk>'])] = 1

        return imgPath, question, label


def collate4VQA4BertViT(batchData):
    labels = torch.stack([data[2] for data in batchData])
    texts = [data[1] for data in batchData]
    imgPaths = [data[0] for data in batchData]
    images = []

    for imgPath in imgPaths:
        images.append(transform(Image.open(imgPath)))

    all = bertTokenizer(texts, return_tensors='pt', padding=True, max_length=512, truncation=True)
    questions = all['input_ids']
    pads = all['attention_mask']

    images = torch.stack(images)

    return images, questions, pads, labels







