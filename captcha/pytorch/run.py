import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from captcha_dataset import CaptchaData
from model import (Model, CNN, VGG)


def evaluate(output, target):
    # one char per row
    output, target = output.view(-1, 26), target.view(-1, 26)
    output = nn.functional.softmax(output, dim=1)
    output = torch.argmax(output, dim=1)
    target = torch.argmax(target, dim=1)

    # 4 char per row
    output, target = output.view(-1, 4), target.view(-1, 4)
    evaluation = [1 if torch.equal(i, j) else 0 for i, j in zip(target, output)]
    return sum(evaluation) / len(evaluation)


if __name__=='__main__':

    # 1 create vgg-like net
    image_size = (3, 60, 120)
    num_classes = 26*4
    conv_pattern = ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512))
    hidden_pattern = (2, 512)

    net = VGG(image_size, num_classes, conv_pattern, hidden_pattern, dropout=0.5)
    # net = CNN()
    num_params = sum(p.numel() for p in net.parameters())

    print(net)  # net architecture
    print(f"Total parameters: {num_params}") # total parameters

    # 2 load data
    train_dataset = CaptchaData('../samples/qq', train=True)
    test_dataset = CaptchaData('../samples/qq', train=False)

    # batch train/test data
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)

    # 3 train and evaluate model
    fun_loss = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.02) # SGD Optimizer

    filename = 'model.pt'
    model = Model(model=net, optimizer=optimizer, loss=fun_loss, stored_model_filename=filename)
    model.fit(
        epochs=20, 
        train_data=train_loader, 
        validate_data=test_loader, 
        fun_evaluate=evaluate,
        output_step=100,
        store_model_filename=filename)