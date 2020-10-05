import torch
import json
import numpy as np
from torch import nn
import pickle

class NCC(nn.Module):
    def __init__(self, trainSplitRatio=0.7, sizeEmbLayer=100, sizeClassfLayer=100, dropOutRatio=0.75,):
        super(NCC, self).__init__()
        self.trainSplitRatio = trainSplitRatio
        self.sizeEmbLayer = sizeEmbLayer
        self.sizeClassfLayer = sizeClassfLayer
        self.dropOutRatio = dropOutRatio

        self.e1Linear1 = nn.Linear(2, self.sizeEmbLayer)
        self.e1BatchNorm1 = nn.BatchNorm1d(self.sizeEmbLayer)
        self.e1ReLu1 = nn.ReLU()
        self.e2Drouput1 = nn.Dropout(self.dropOutRatio)
        self.e1Linear2 = nn.Linear(self.sizeEmbLayer, self.sizeEmbLayer)
        self.e1BatchNorm2 = nn.BatchNorm1d(self.sizeEmbLayer)
        self.e1ReLu2 = nn.ReLU()
        self.e2Drouput2 = nn.Dropout(self.dropOutRatio)

        self.classLayer = nn.Sequential(
            nn.Linear(self.sizeEmbLayer, self.sizeClassfLayer),
            nn.BatchNorm1d(self.sizeClassfLayer),
            nn.ReLU(),
            nn.Dropout(self.dropOutRatio),
            nn.Linear(self.sizeClassfLayer, self.sizeClassfLayer),
            nn.BatchNorm1d(self.sizeClassfLayer),
            nn.ReLU(),
            nn.Dropout(self.dropOutRatio),
        )
        self.logits = nn.Linear(self.sizeClassfLayer, 1)

    def forward(self, xVal, yVal):
        xyval = torch.cat([xVal, yVal], 2)
        BatchSize = xyval.shape[0]
        DataSize = xyval.shape[1]
        e1L1 = self.e1Linear1(xyval).view(BatchSize, self.sizeEmbLayer, DataSize)
        e1B1 = self.e1BatchNorm1(e1L1).view(BatchSize, DataSize, self.sizeEmbLayer)
        e1R1 = self.e1ReLu1(e1B1)
        e1D1 = self.e2Drouput1(e1R1)
        e1L2 = self.e1Linear2(e1D1).view(BatchSize, self.sizeEmbLayer, DataSize)
        e1B2 = self.e1BatchNorm2(e1L2).view(BatchSize, DataSize, self.sizeEmbLayer)

        e1R2 = self.e1ReLu2(e1B2)
        e1D2 = self.e2Drouput2(e1R2)


        finalEmbLayer = torch.mean(e1D2, 1)

        classLayer = self.classLayer(finalEmbLayer)
        logits = self.logits(classLayer)
        prob = torch.sigmoid(logits)
        return logits, prob

def returnTorch(listObj):
    XList = []
    YList = []
    LabelList = []
    for obj in listObj:
        tempX = np.array(obj["trainX"])
        tempX = tempX[np.newaxis, ...]
        XList.append(tempX)

        tempY = np.array(obj["trainY"])
        tempY = tempY[np.newaxis, ...]
        YList.append(tempY)

        tempLabel = np.array([obj["label"]])
        tempLabel = tempLabel[np.newaxis, ...]
        LabelList.append(tempLabel)

    ## format the data into np.float32 and single array
    X = np.concatenate(XList, axis=0)
    Y = np.concatenate(YList, axis=0)
    Label = np.concatenate(LabelList, axis=0)
    X = X[..., np.newaxis]
    Y = Y[..., np.newaxis]
    return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(Label)

def returnTorchForVector(listObj):
    XList = []
    YList = []
    for obj in listObj:
        tempX = np.array(obj["trainX"])
        tempX = tempX[np.newaxis, ...]
        XList.append(tempX)

        tempY = np.array(obj["trainY"])
        tempY = tempY[np.newaxis, ...]
        YList.append(tempY)

    X = np.concatenate(XList, axis=0)
    Y = np.concatenate(YList, axis=0)
    X = X[..., np.newaxis]
    Y = Y[..., np.newaxis]
    return torch.from_numpy(X), torch.from_numpy(Y)

def testNCC():
    tubDataset = "./data/tubehengenDataFormat.json"
    model = torch.load('./model/NCC_model_final.pt')
    model.eval()
    with torch.no_grad():
        with open(tubDataset, "r") as tubDataReader:
            count = 0
            correct = 0
            for line in tubDataReader:
                data = json.loads(line)
                testInputX, testInputY, testLabel = returnTorch([data])
                _, prob = model(testInputX.cuda().float(), testInputY.cuda().float())

                if prob[0][0] > (1 - prob[0][0]):
                    prediction = 1
                else:
                    prediction = 0
                count += 1

                if prediction == testLabel[0][0]:
                    correct += 1
                else:
                    print("wrong Prediction : prob : %f label : %f" % (prob[0][0], testLabel[0][0]))

                print("count : ", count, "correct : ", correct)
            print("accuracy: ", correct / count)
            return correct / count

classLabel = {'person': 0, 'chair': 1, 'car': 2, 'dog': 3, 'bottle': 4, 'cat': 5, 'bird': 6,
                           'pottedplant': 7, 'sheep': 8, 'boat': 9, 'aeroplane': 10, 'tvmonitor': 11, 'sofa': 12,
                           'bicycle': 13, 'horse': 14, 'motorbike': 15, 'diningtable': 16, 'cow': 17, 'train': 18,
                           'bus': 19}

def ExtractDataForVector():
    FileName = "resnetModelFeatureVector.json"
    Dataset = []
    with open(FileName, "r") as DataReader:
        for line in DataReader:
            data = json.loads(line)
            Dataset.append(data)

    Dataset = np.array(Dataset)
    namelist = np.array([s["className"] for s in Dataset])
    NCCjsonDict = {}
    for className in classLabel.keys():
        classdata = Dataset[namelist == className]
        if classdata.any():
            TrainX, TrainY = returnTorchForVector(classdata)
            NCCjsonDict[className] = {"TrainX": TrainX, "TrainY": TrainY}
        else:
            continue
    return NCCjsonDict

def testForVector():
    NCCjsonDict = ExtractDataForVector()
    CausalDict = {}
    model = torch.load('./model/NCC_model_final.pt')
    model.eval()
    with torch.no_grad():
        for className, data in NCCjsonDict.items():
            print(className, "...")
            InputX = data["TrainX"]
            InputY = data["TrainY"]
            print(InputX.shape)
            _, prob = model(InputX.cuda().float(), InputY.cuda().float())
            CausalDict[className] = prob.data.cpu().numpy()
    with open("./CausalDict.pickle", "wb") as fp:
        pickle.dump(CausalDict, fp)

if __name__ == '__main__':
    testForVector()
