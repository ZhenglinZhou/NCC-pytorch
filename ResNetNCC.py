import numpy as np
import os
import torch
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
import cv2
from bs4 import BeautifulSoup
from torchvision import transforms, models
from torch import nn, optim
from MyResNetModelTraining import GenNCC
from tensorboardX import SummaryWriter

class VocDataset(Dataset):
    def __init__(self, transform=None):
        self.annotateDir = "/media/zhou/data/VOC/VOCdevkit/VOC2012/Annotations"
        self.imageDir = "/media/zhou/data/VOC/VOCdevkit/VOC2012/JPEGImages"
        self.vocClass = 20
        self.classLabel = {'person': 0, 'chair': 1, 'car': 2, 'dog': 3, 'bottle': 4, 'cat': 5, 'bird': 6,
                           'pottedplant': 7, 'sheep': 8, 'boat': 9, 'aeroplane': 10, 'tvmonitor': 11, 'sofa': 12,
                           'bicycle': 13, 'horse': 14, 'motorbike': 15, 'diningtable': 16, 'cow': 17, 'train': 18,
                           'bus': 19}
        self.antiLabel = dict([(value, key) for key, value in self.classLabel.items()])
        self.transform = transform
        self.loadAnnotationFile()

    def __len__(self):
        return len(self.imageList)

    def __getitem__(self, index):
        annot = self.annotations[self.imageList[index]]
        img = self.load_image(self.imageList[index])
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, filename):
        image_path = os.path.join(self.imageDir, filename)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32) / 255.0

    def loadAnnotationFile(self):
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
        self.annotations = dataset
        self.imageList = list(dataset.keys())

class Normalizer(object):
    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        image, annot = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annot}

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
    padded_annots = torch.zeros(batch_size, 20)
    for i in range(batch_size):
        annot = annots[i]
        padded_annots[i, :] = annot
    """
        output
        img = [batch_size x 3 x W x H]
    """
    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'annot': padded_annots}

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
        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots)}

class NCCResNet():
    def __init__(self):
        self.LR = 0.001
        self.MAX_EPOCH = 5
        self.splitRatio = 0.75
        self.batchSize = 32
        self.BestAcc = 0

        ''' ############### ResNet ############### '''
        self.resnet_model = models.resnet50(pretrained=True).cuda()
        self.resnet_model.fc = nn.Sequential()
        for param in self.resnet_model.parameters():
            param.requires_grad = False

        ''' ############### MyModel ############### '''
        self.model = GenNCC().cuda()
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.LR, momentum=0.9, weight_decay=5e-4)
        # ExpLR = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        ''' ############### DataLoad ############### '''
        transform_list = [Normalizer(), Resizer()]
        self.dataset = VocDataset(transform=transforms.Compose(transform_list))
        self.train_sampler, self.test_sampler = self.make_sampler(self.dataset)
        self.train_loader = DataLoader(self.dataset, batch_size=self.batchSize, sampler=self.train_sampler, collate_fn=collater)
        self.test_loader = DataLoader(self.dataset, batch_size=self.batchSize, sampler=self.test_sampler, collate_fn=collater)

        ''' ############### TensorBoard ############### '''
        self.writer = SummaryWriter('runs/ResNetCNN')

    def make_sampler(self, dataset):
        indices = list(range(len(dataset)))
        split = int(np.floor(self.splitRatio * len(dataset)))
        np.random.shuffle(indices)
        train_indices, test_indices = indices[:split], indices[split:]
        return SubsetRandomSampler(train_indices), SubsetRandomSampler(test_indices)

    def Train(self):
        self.model.train()
        count = 0
        for epoch in range(self.MAX_EPOCH):
            self.model.train()
            for iter, data in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                count += 1
                img = torch.Tensor(data['img']).cuda()
                annots = torch.Tensor(data['annot']).cuda()

                features = self.resnet_model(img)
                _, logits, prob = self.model(features)
                loss = self.criterion(logits, annots)
                loss.backward()
                self.optimizer.step()
                # ExpLR.step()
                self.writer.add_scalar('Train', loss.item(), count)
                print("epoch:", epoch, "iter:", iter, "loss:", loss.item())

            self.Test(epoch)
            # if (epoch % 2 == 0):
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] *= 0.1

    def Test(self, epoch):
        with torch.no_grad():
            self.model.eval()
            correct = 0
            datasize = 0
            loss_list = []
            for iter, data in enumerate(self.test_loader):
                img = torch.Tensor(data['img']).cuda()
                annots = torch.Tensor(data['annot']).cuda()
                features = self.resnet_model(img)
                _, logits, prob = self.model(features)
                loss = self.criterion(logits, annots)
                loss_list.append(loss.item())
                correct_counts = torch.eq(torch.round(prob), annots)
                correct += torch.sum(correct_counts).item()
                # correct += torch.sum(torch.round(prob)[correct_counts]).item()
                datasize += torch.numel(annots)
            acc = correct / datasize
            if acc > self.BestACC:
                self.BestACC = acc
                torch.save(self.model, 'ResNet_model_final.pt')
            self.writer.add_scalar("Test Loss", np.mean(loss_list), epoch)
            self.writer.add_scalar("Test Acc", acc, epoch)
            print("epoch:", epoch, "acc:", acc, "loss:", np.mean(loss_list))



if __name__ == '__main__':
    # save model named ResNet_model_final.pt
    obj = NCCResNet()
    obj.Train()




