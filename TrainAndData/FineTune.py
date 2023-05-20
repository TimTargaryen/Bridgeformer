import sys

sys.path.append("..")
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from torch.utils.tensorboard import SummaryWriter

import yaml
import datetime
import os


#config = yaml.load(open("../config.yaml"), yaml.FullLoader)
#beginTime = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")


def line(c):
    for i in range(50):
        print(c, end="")
    print()


def getRate(model, LR, decay):
    rate = [{'params': model.BertBackbone.encoder.layer[i].parameters(),
             'lr': LR * (decay ** (11 - i))} for i in range(12)]
    rate.append({'params': model.BertBackbone.embeddings.parameters(), 'lr': 0})
    return rate


def VqaFineTune(model, trainLoader, valLoader, epochs, LR):
    writer = SummaryWriter(os.path.join(config['saveDir'], beginTime))
    citerition = nn.BCELoss()

    interval = (config['Interval'] / trainLoader.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    show = 1

    for epoch in range(epochs):

        cnt = 0
        Loss = 0.0
        correct = 0.0

        #rate = getRate(model, LR, 0.95)
        model.train()
        model.freeze()
        for i, (img, seq, pad, label) in enumerate(trainLoader):
            cnt += 1

            img = img.to(device)
            seq = seq.to(device)
            pad = pad.to(device)
            label = label.to(device)

            predict = model(img, seq, pad)

            loss = citerition(predict, label)
            loss = loss.mean()
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            Loss += loss.item()

            pred = torch.zeros((len(img), len(label[0]))) ## multi label
            pred[:, :] = 2
            pred[torch.arange(predict.shape[0]), torch.argmax(predict, dim=-1)] = 1
            pred[:, 0] = 2
            correct += (torch.sum(pred == label) / len(label)).item()

            if loss.item() > 20000 or Loss != Loss:
                print(loss)
                print("sth wrong")
                exit(1)

            if cnt % interval == 0 and cnt != 0:
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                print("epoch:{}/{}, step:{}/{}, avgloss:{}, correctRate:{}"
                      .format(epoch + 1, epochs, cnt,
                              trainLoader.__len__(), Loss / interval, correct / interval, lr))

                writer.add_scalar('avgloss', Loss / interval, epoch * trainLoader.__len__() + cnt)
                writer.add_scalar('correctRate', correct / interval, epoch * trainLoader.__len__() + cnt)
                Loss = 0
                correct = 0.0
                break

        if (epoch + 1) % show == 0 and epoch > 0:
            cnt = 0

            with torch.no_grad():
                for i, (img, seq, pad, label) in enumerate(valLoader):
                    cnt += 1

                    img = img.to(device)
                    seq = seq.to(device)
                    pad = pad.to(device)
                    label = label.to(device)

                    predict = model(img, seq, pad)
                    pred = torch.zeros((len(img), len(label[0])))  ## multi label
                    pred[:, :] = 2
                    pred[torch.arange(predict.shape[0]), torch.argmax(predict, dim=-1)] = 1
                    pred[:, 0] = 2
                    correct += (torch.sum(pred == label) / len(label)).item()

                    if cnt % interval == 0 and cnt != 0:
                        line("-")
                        print("epoch:{}/{}, step:{}/{}, AvgCorrectRate:{}"
                              .format(epoch + 1, epochs, cnt,
                                      trainLoader.__len__(), correct / cnt))

                line("*")
                line("*")
                print("epoch:{}/{}, testCorrectRate:{}".format(epoch + 1, epochs, correct / cnt))
                line("*")
                line("*")

            writer.add_scalar('testCorrectRate', correct / cnt, epoch + 1)
            torch.save(model, os.path.join(config['saveDir'], beginTime, str(epoch + 1) + ".pth"))

if __name__ == "__main__":
    from BridgeFormer.model import ViT4CLS, Bert4CLS, Bridge

    vit = ViT4CLS(torch.load("../vit.pth"))
    bert = Bert4CLS(torch.load("../bert.pth"))
    bridge = Bridge(vit, bert)

    from VQAv2 import VQAv2, collate4VQA4BertViT

    v2 = VQAv2("F:\\datasets\\MultiModal\\VQAv2")
    VQAtrain = DataLoader(v2, batch_size=4, collate_fn=collate4VQA4BertViT)

    VqaFineTune(bridge, VQAtrain, VQAtrain, 1, 1e-5)
