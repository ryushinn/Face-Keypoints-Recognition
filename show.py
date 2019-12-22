import cv2
import torch
from os import path
from dataloader import *
from torchvision import transforms


def pred():
    img_id = 35
    img = cv2.imread(f"dataset/test/{img_id:04}.jpg")
    f = open("result/submission.csv")
    points = []
    values = f.readlines()[img_id + 1].split(",")
    for i in range(98):
        points.append((int(float(values[2 * i + 1])), int(float(values[2 * i + 2]))))

    for p in points:
        cv2.circle(img, p, 1, (0, 255, 0), 2)
    cv2.imshow("test", img)
    cv2.waitKey(0)


def eval():
    load_path = path.join(path.dirname(__file__), "model", "model_change_kernel_size.pth")
    net = torch.load(load_path)
    net.eval()
    img_id = 9
    img = cv2.imread(f"dataset/train/{img_id:04}.jpg")
    img_new, ratio = rescale(img, (200, 200))

    # img_new = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)[:, :, None]
    print(img_new.shape)
    with torch.no_grad():
        outputs = net(transforms.ToTensor()(img_new).view(-1, 3, 200, 200))

    points = []
    for i in range(98):
        points.append((int(float(outputs[0][2 * i]/ratio)), int(float(outputs[0][2 * i + 1]/ratio))))
    for p in points:
        cv2.circle(img, p, 1, (0, 255, 0), 2)

    pts = open(f"dataset/train/{img_id:04}.pts")
    points = []
    for i, line in enumerate(pts.readlines()):
        point = line.split(",")
        # keypoints rescale
        points.append((int(float(point[0])), int(float(point[1]))))
    for p in points:
        cv2.circle(img, p, 1, (255, 0, 0), 2)
    cv2.imshow("test", img)
    cv2.waitKey(0)


def origin():
    img_id = 0
    img = cv2.imread(f"dataset/train/{img_id:04}.jpg")
    pts = open(f"dataset/train/{img_id:04}.pts")
    points = []
    for i, line in enumerate(pts.readlines()):
        point = line.split(",")
        # keypoints rescale
        points.append((int(float(point[0])), int(float(point[1]))))
    for p in points:
        cv2.circle(img, p, 1, (255, 0, 0), 2)
    cv2.imshow("test", img)
    cv2.waitKey(0)
    pass


if __name__ == '__main__':
    #pred()
    #origin()
    eval()
