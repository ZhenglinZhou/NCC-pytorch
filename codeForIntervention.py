import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from bs4 import BeautifulSoup
from PIL import Image
import torch
from torch import nn
from torchvision import models

classMap = {'person': 0, 'chair': 1, 'car': 2, 'dog': 3, 'bottle': 4, 'cat': 5, 'bird': 6, 'pottedplant': 7, 'sheep': 8, 'boat': 9, 'aeroplane': 10, 'tvmonitor': 11, 'sofa': 12, 'bicycle': 13, 'horse': 14, 'motorbike': 15, 'diningtable': 16, 'cow': 17, 'train': 18, 'bus': 19}

def loadImageData(xmlFolderPath):
    ## function to load image from given folder for testing the model
    dictOfImage = {}
    limit = 10000

    for root, dirs, files in os.walk(xmlFolderPath):
        fileCounter = 0

        for elm in files[:limit]:
            fileName = os.path.join(xmlFolderPath, elm)
            with open(fileName, "r") as f:
                xml = f.readlines()
                xml = ''.join([line.strip('\t') for line in xml])
                annXml = BeautifulSoup(xml)
                fileCounter += 1
                print("fileLoaded : %d " % (fileCounter))
                objs = annXml.findAll('object')
                img = annXml.filename.string

                # imagePt = Image.open(fullPath)

                # imgArr = np.asarray(imagePt)
                # newImgArr = imgArr.copy()
                # orignalImage = imgArr.copy()
                # contextImg = imgArr.copy()

                for obj in objs:
                    obj_names = obj.findChildren('name')
                    bndBox = obj.findChildren('bndbox')
                    for classLabel, box in zip(obj_names, bndBox):
                        label = classLabel.contents[0]
                        minX = int(float(box.findChildren('xmin')[0].contents[0]))
                        minY = int(float(box.findChildren('ymin')[0].contents[0]))
                        maxX = int(float(box.findChildren('xmax')[0].contents[0]))
                        maxY = int(float(box.findChildren('ymax')[0].contents[0]))
                        # if label in classMap:
                        # 	newImgArr[minY:maxY,minX:maxX,:] = 0
                        # else:
                        # 	pass

                        if label in classMap:
                            if label not in dictOfImage:
                                dictOfImage[label] = {img: [(minY, maxY, minX, maxX)]}
                            elif (label in dictOfImage) and (img in dictOfImage[label]):
                                dictOfImage[label][img].append((minY, maxY, minX, maxX))
                            elif (label in dictOfImage) and (img not in dictOfImage[label]):
                                dictOfImage[label][img] = [(minY, maxY, minX, maxX)]

                            else:
                                pass
                        else:
                            pass
            # contextImg[newImgArr!=0] =0
            # newImg = Image.fromarray(contextImg)
            # plt.imshow(newImg)

            # plt.show()
    return dictOfImage

def boxPlot(feat, orignalFeat, antiCasIdx, casIdx):
    ## drawing the box plot for a given class

    featCasList = []
    featAntiCasList = []

    ############### calculating for object feature vector #############
    for index in casIdx:
        Num = 0
        Den = 0
        for obF1, origF1 in zip(feat, orignalFeat):
            Num += np.abs(obF1[index] - origF1[index])
            Den += np.abs(origF1[index])

        featCasList.append(Num / Den)

    ############### calculating for object feature vector #############
    for index in antiCasIdx:
        Num = 0
        Den = 0
        for obF1, origF1 in zip(feat, orignalFeat):
            Num += np.abs(obF1[index] - origF1[index])
            Den += np.abs(origF1[index])

        featAntiCasList.append(Num / Den)

    return featCasList, featAntiCasList

def imageVectorSample(imageFolderPath, dictOfImage, className):
    ### model initialization
    resnet_model = models.resnet50(pretrained=True).cuda()
    resnet_model.fc = nn.Sequential()
    for param in resnet_model.parameters():
        param.requires_grad = False

    ''' ############### MyModel ############### '''
    model = torch.load('./ResNet_model_final.pt').cuda()

    imageForClass = dictOfImage[className]

    objImgFeat = []
    origImgFeat = []
    contImgFeat = []

    for image in imageForClass:
        fullPath = os.path.join(imageFolderPath, image)
        imagePt = Image.open(fullPath)
        imagePt = imagePt.resize((300, 300))
        imgArr = np.asarray(imagePt)
        objImg = imgArr.copy()
        orignalImage = imgArr.copy()
        contextImg = imgArr.copy()

        ## iterating over the list value
        for minY, maxY, minX, maxX in imageForClass[image]:
            contextImg[minY:maxY, minX:maxX, :] = 0

        objImg[contextImg != 0] = 0



        imageFrames = np.concatenate(
            [orignalImage[np.newaxis, ...], objImg[np.newaxis, ...], contextImg[np.newaxis, ...]], axis=0)
        imageFrames = torch.from_numpy(imageFrames)
        imageFrames = imageFrames.permute(0, 3, 1, 2)

        ResNetFeatures = resnet_model(imageFrames.cuda().float())
        NCCFeatures, logits, _ = model(ResNetFeatures)
        NCCFeatures = NCCFeatures.data.cpu()
        origImgFeat.append(NCCFeatures[0])
        objImgFeat.append(NCCFeatures[1])
        contImgFeat.append(NCCFeatures[2])

    return origImgFeat, objImgFeat, contImgFeat


def loadImageClass(className):
    with open("CausalDict.pickle", 'rb') as pickleFileReader:
        CausalDict = pickle.load(pickleFileReader)
    dataBasedClass = CausalDict[className]
    sortidx = np.argsort(dataBasedClass, axis=0)
    casidxList = sortidx[:100]
    antiCasidxList = sortidx[-100:]
    return antiCasidxList, casidxList



if __name__ == "__main__":
    outDict = loadImageData(xmlFolderPath="/media/zhou/data/VOC/VOCdevkit/VOC2012/Annotations")
    # labels = ['pottedplant', 'bicycle', 'car', 'train', 'chair', 'tvmonitor', 'sheep', 'cow', 'person', 'diningtable', 'horse',
    #  'motorbike', 'cat', 'bus', 'sofa', 'bird', 'dog', 'bottle', 'boat', 'aeroplane']
    labels = ['person', 'chair', 'car', 'dog', 'bottle', 'cat', 'bird', 'pottedplant', 'sheep', 'boat', 'aeroplane', 'tvmonitor',
     'sofa', 'bicycle', 'horse', 'motorbike', 'diningtable', 'cow', 'train']
    meanCas = []
    stdCas = []
    meanAnti = []
    stdAnti = []

    for label in labels:
        print(label, "...")
        origImgFeat, objImgFeat, contImgFeat = imageVectorSample(
            imageFolderPath="/media/zhou/data/VOC/VOCdevkit/VOC2012/JPEGImages", dictOfImage=outDict, className=label)
        antiCasidxList, casidxList = loadImageClass(label)
        featCas, featAntiCas = boxPlot(feat=contImgFeat, orignalFeat=origImgFeat, antiCasIdx=antiCasidxList,
                                       casIdx=casidxList)

        meanCas.append(np.mean(featCas))
        stdCas.append(np.std(featCas))

        meanAnti.append(np.mean(featAntiCas))
        stdAnti.append(np.std(featAntiCas))
    tick_label = labels
    x = np.arange(len(tick_label))

    # fig, ax = plt.subplots()
    bar_width = 0.35
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    plt.bar(x, meanAnti, bar_width,
            alpha=opacity, color='r',
            yerr=stdAnti, error_kw=error_config,
            label='AntiCasual')
    plt.bar(x+bar_width, meanCas, bar_width,
                    alpha=opacity, color='b',
                    yerr=stdCas, error_kw=error_config,
                    label='Casual', align="center")
    plt.xticks(x+bar_width/2, tick_label, rotation=45)
    plt.legend()
    plt.show()
