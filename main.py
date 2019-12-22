from dataloader import *
from os import path
from recog_model import FaceRecognition
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader


def train(train_loader, save_path=None):
    print("======TRAIN======")
    net = FaceRecognition()

    if torch.cuda.is_available():
        net = net.cuda()

    criterion = nn.MSELoss(reduction="sum")
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    net.train()
    for epoch in range(epoch_num):
        avg_loss = 0.0
        show_per_batch = 50
        for i, data in enumerate(train_loader):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            avg_loss += (loss.item() / outputs.size().numel())
            if (i + 1) % show_per_batch == 0:
                print(f"epoch {epoch + 1} batch {i + 1}: the avg_loss is {avg_loss / show_per_batch:.3f}")
                avg_loss = 0.0
    if save_path is not None:
        torch.save(net.cpu(), save_path)


def evaluate(val_loader, load_path):
    net = torch.load(load_path)
    if torch.cuda.is_available():
        net = net.cuda()

    total_loss = 0.0
    criterion = nn.MSELoss(reduction="sum")
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    print("======EVAL======")
    print(f"the total loss is {total_loss:3f}")
    print(f"the mean loss is {total_loss / testset_size:3f}")
    print(f"loss in every coordinate is {total_loss / (testset_size * 98 * 2):3f}")


def predict(test_loader, load_path):
    print("======PREDICT======")
    net = torch.load(load_path)
    if torch.cuda.is_available():
        net = net.cuda()

    net.eval()
    result_path = path.join(path.dirname(__file__), "result", "submission_change_kernel_size.csv")
    res = open(result_path, "w")
    res.write("id,landmarks\n")
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, scale_ratio = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                scale_ratio = scale_ratio.cuda()
            outputs = net(inputs)

            outputs = outputs.view(-1)
            scale_ratio = scale_ratio.view(-1)
            outputs /= scale_ratio

            res.write(f"{i}")
            for j in range(98):
                res.write(f",{(outputs[2 * j]).item():6f},{(outputs[2 * j + 1]).item():6f}")
            res.write("\n")


trainset_size = 6000
valset_size = 1500
testset_size = 1000
epoch_num = 5


def main():
    False
    True
    train_ = False
    eval_ = False
    predict_ = True
    model_path = path.join(path.dirname(__file__), "model", "model_change_kernel_size.pth")
    labeled_imgs_path = [path.join(path.dirname(__file__), "dataset", "train", f"{i:04}.jpg")
                         for i in range(trainset_size + valset_size)]
    labels_path = [path.join(path.dirname(__file__), "dataset", "train", f"{i:04}.pts")
                   for i in range(trainset_size + valset_size)]
    unlabeled_imgs_path = [path.join(path.dirname(__file__), "dataset", "test", f"{i:04}.jpg")
                           for i in range(testset_size)]
    trainset, valset, testset = load_data(labeled_imgs_path, labels_path, unlabeled_imgs_path)

    train_loader = DataLoader(trainset, batch_size=8, shuffle=True)
    val_loader = DataLoader(valset, batch_size=4, shuffle=True)
    test_loader = DataLoader(testset, batch_size=1, shuffle=False)

    print(f"CUDA? {torch.cuda.is_available()}")
    if train_:
        train(train_loader, model_path)
    if eval_:
        evaluate(val_loader, model_path)
    if predict_:
        predict(test_loader, model_path)


if __name__ == "__main__":
    main()
