import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from bs4 import BeautifulSoup
import random
from torchvision import transforms
import json
from ResNetNCC import NCCResNet
from ResNetNCC import GenNCC


class createDataNCCTest(Dataset):
    def __init__(self, batchSize=16, transform=None):
        self.batchSize = batchSize
        self.vocClass = 20
        self.noElmPerclass = self.batchSize * 50
        self.annotateDir = "/media/zhou/data/VOC/VOCdevkit/VOC2012/Annotations"
        self.imageDir = "/media/zhou/data/VOC/VOCdevkit/VOC2012/JPEGImages"
        self.classLabel = {'person': 0, 'chair': 1, 'car': 2, 'dog': 3, 'bottle': 4, 'cat': 5, 'bird': 6,
                           'pottedplant': 7, 'sheep': 8, 'boat': 9, 'aeroplane': 10, 'tvmonitor': 11, 'sofa': 12,
                           'bicycle': 13, 'horse': 14, 'motorbike': 15, 'diningtable': 16, 'cow': 17, 'train': 18,
                           'bus': 19}
        self.antiLabel = dict([(value, key) for key, value in self.classLabel.items()])
        self.transform = transform

        self.data = self.loadAnnotationFile()
        self.imageList = list(self.data.keys())
        random.shuffle(self.imageList)
        self.classDict = self.CreateClassDict()
        self.RefinClassDict()

        self.imageList = []
        self.imageClassNamelist = []
        for className, file in self.classDict.items():
            self.imageClassNamelist.extend(np.tile(className, len(file)))
            self.imageList.extend(file)

    def __getitem__(self, item):
        fileName = self.imageList[item]
        className = self.imageClassNamelist[item]
        img = self.load_image(fileName)
        sample = {"img": img, "annot": className}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.imageList)

    def load_image(self, filename):
        image_path = os.path.join(self.imageDir, filename)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32) / 255.0

    def RefinClassDict(self):
        # make sure len(dataPerClass) % batchSize = 0
        for className, dataPerClass in self.classDict.items():
            idxLt = (len(dataPerClass) // self.batchSize) * self.batchSize
            dataPerClass = dataPerClass[:idxLt]
            self.classDict[className] = dataPerClass

    def CreateClassDict(self):
        # classDict["ClassName"] = [fileName1, fileName2, ...]
        classDict = {}
        for image in self.imageList:
            classPerImage = self.data[image]
            for idx, val in enumerate(classPerImage):
                if val == 1:
                    if self.antiLabel[idx] not in classDict:
                        classDict[self.antiLabel[idx]] = [image]
                    else:
                        if (len(classDict[self.antiLabel[idx]]) < self.noElmPerclass):
                            classDict[self.antiLabel[idx]].append(image)
                        else:
                            pass
                else:
                    continue
        return classDict

    def loadAnnotationFile(self):
        # dataset = [fileName1, fileName2, ...]
        dataset = {}
        fileCounter = 0
        for root, dirs, files in os.walk(self.annotateDir):
            for elm in files:
                fileName = os.path.join(self.annotateDir, elm)
                with open(fileName, "r") as f:
                    xml = f.readlines()
                    xml = ''.join([line.strip('\t') for line in xml])
                    annXml = BeautifulSoup(xml)
                    oneHotEncode = [0] * self.vocClass
                    fileCounter += 1
                    print("fileLoaded : %d " % (fileCounter))
                    objs = annXml.findAll('object')
                    for obj in objs:
                        obj_names = obj.findChildren('name')
                        for nameTag in obj_names:
                            if nameTag.contents[0] in self.classLabel:
                                oneHotEncode[self.classLabel[nameTag.contents[0]]] = 1
                                # Encode = self.classLabel[nameTag.contents[0]]
                            else:
                                continue
                    dataset[annXml.filename.string] = np.array(oneHotEncode).astype(np.float32)
                    # dataset[annXml.filename.string] = Encode
        return dataset


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]

    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)
    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img
    """
        output
        img = [batch_size x 3 x W x H]
    """
    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'annot': annots}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, min_side=300, max_side=300):  # 608-1024
        image, annots = sample['img'], sample['annot']

        rows, cols, cns = image.shape
        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        """
            image = [H * W * 3]
            cv2.resize(image, (resize_W, resize_H))
        """
        image = cv2.resize(image, (int(round(cols * scale)), int(round((rows * scale)))))
        rows, cols, cns = image.shape
        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32
        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)
        return {'img': torch.from_numpy(new_image), 'annot': annots}


class GenResNetNCCVector():
    def __init__(self):
        self.batchSize = 16
        self.ResNetNCC = NCCResNet()
        transform_list = [Resizer()]
        self.dataset = createDataNCCTest(batchSize=self.batchSize, transform=transforms.Compose(transform_list))
        self.dataLoader = DataLoader(self.dataset, batch_size=self.batchSize, collate_fn=collater)

    def GenerateFeatureLabelVector(self):
        with open("./data/resnetModelFeatureVector.json", "w") as fetVectorWriter:
            labelName = None
            for iter, data in enumerate(self.dataLoader):
                img = torch.Tensor(data['img']).cuda()
                annots = data['annot']
                NCCFeatures, logits = self.ResNetNCC.TestOtherFormat(img)
                className = annots[0]
                classId = self.dataset.classLabel[className]
                trainY = logits[:, classId].data.cpu()

                if className != labelName:
                    if labelName != None:
                        print(labelName, "have done.")
                        for elm in featureLogitDict:
                            json.dump(featureLogitDict[elm], fetVectorWriter)
                            fetVectorWriter.write("\n")
                    labelName = className
                    print(className, "will be done...")
                    featureLogitDict = {}
                    for idx in range(512):
                        featureLogitDict[idx] = {"trainX": [], "trainY": [], "className": className,
                                                 "featureIdx": idx, "label": 0}

                for idx in range(512):
                    featureLogitDict[idx]["trainY"].extend(trainY.numpy().ravel().tolist())
                    featureLogitDict[idx]["trainX"].extend(NCCFeatures[:, idx].numpy().ravel().tolist())


if __name__ == '__main__':
    obj = GenResNetNCCVector()
    obj.GenerateFeatureLabelVector()
