import cv2
import torch
import torch.utils.data
from main import trainset_size, valset_size
from torchvision import transforms


class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y=None, ratios=None, transformer=transforms.ToTensor()):
        self.x = x
        self.y = y
        self.ratios = ratios
        self.t = transformer

    def __getitem__(self, idx):
        if self.y is not None:
            return self.t(self.x[idx]), torch.tensor(self.y[idx])
        elif self.ratios is not None:
            return self.t(self.x[idx]), torch.tensor(self.ratios[idx])

    def __len__(self):
        return len(self.x)


def load_data(labeled, labels, unlabeled):
    target_size = (200, 200)
    trainset = []
    keypoints = []
    testset = []
    ratios = []
    for img_path, label_path in zip(labeled, labels):
        img = cv2.imread(img_path)
        img, ratio = rescale(img, target_size)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:, :, None]
        pts = open(label_path)
        points = []
        for i, line in enumerate(pts.readlines()):
            point = line.split(",")
            # keypoints rescale
            points.append(float(point[0]) * ratio)
            points.append(float(point[1]) * ratio)
        keypoints.append(points)
        trainset.append(img)
    for img_path in unlabeled:
        img = cv2.imread(img_path)
        img, ratio = rescale(img, target_size)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:, :, None]
        testset.append(img)
        ratios.append(ratio)

    trainset = Dataset(trainset, y=keypoints)
    testset = Dataset(testset, ratios=ratios)

    trainset, valset = torch.utils.data.random_split(trainset, [trainset_size, valset_size])

    return trainset, valset, testset


def rescale(img, target_size):
    """
    :param img: 输入图片
    :param target_size: 目标size
    :return: 等比例缩放到目标size(左上角重合)，不足用0填充，返回缩放后的图片和缩放比例ratio
    """
    old_size = img.shape[0:2]

    ratio = min([target_size[i] / old_size[i] for i in range(len(old_size))])
    new_size = [int(old_size[i]*ratio) for i in range(len(old_size))]
    bottom = target_size[0] - new_size[0]
    right = target_size[1] - new_size[1]

    img = cv2.resize(img, (new_size[1], new_size[0]))
    img = cv2.copyMakeBorder(img, 0, bottom, 0, right, cv2.BORDER_CONSTANT, None, (0, 0, 0))

    return img, ratio


if __name__ == "__main__":
    img = cv2.imread("dataset/test/0000.jpg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(img.shape)
    print(img_gray.shape)
    cv2.imshow("RGB", img)
    cv2.waitKey(0)
    cv2.imshow("GRAY", img_gray)
    cv2.waitKey(0)
