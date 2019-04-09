'''
Code written by: Xiaoqing Liu
If you use significant portions of this code or the ideas from our paper, please cite it :)
'''
import argparse
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import os
from danet import DANet
import torch

from torch.nn import functional as F
import numpy as np
from utils import poly_lr_scheduler
from utils import reverse_one_hot, get_label_info, colour_code_segmentation, compute_global_accuracy

from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad
from torchvision.transforms import ToTensor, ToPILImage
from dataset import train,test
from transform import Relabel, ToLabel, Colorize
from customize import SegmentationMultiLosses

image_transform = ToPILImage()
input_transform = Compose([
    Resize((576,576)),
    ToTensor(),
])
target_transform = Compose([
    Resize((576,576)),
    ToLabel(),
])


def train(args, model, optimizer, dataloader_train):

    step = 0
    for epoch in range(1,args.num_epochs):
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        model.train()

        loss_record = []
        for i,(data, label) in enumerate(dataloader_train):

            data = data.cuda()
            label = label.cuda()
            output = model(data)
            print type(output)
            criterion = SegmentationMultiLosses(nclass=2).cuda()

            loss = criterion(output, label[:,0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            loss_record.append(loss.item())
            if i%50 == 0:
                loss_averge = sum(loss_record)/len(loss_record)
                print ('epoch:%f'%epoch,'step:%f'%i,'loss:%f'%loss_averge)
                #torch.save(model.state_dict(), os.path.join(args.save_model_path, 'epoch_{}.pth'.format(epoch)))

        loss_train_mean = np.mean(loss_record)

        print('loss for train : %f' % (loss_train_mean))
        if epoch % args.checkpoint_step == 0:
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save(model.state_dict(), os.path.join(args.save_model_path, 'epoch_{}.pth'.format(epoch)))



def main(params):

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
    parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--checkpoint_step', type=int, default=5, help='How often to save checkpoints (epochs)')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate used for train')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--save_model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')

    args = parser.parse_args(params)

    # create dataset and dataloader
    dataloader_train = DataLoader(train(input_transform, target_transform),num_workers=1, batch_size=2, shuffle=True)

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = DANet(nclass=2, backbone='resnet50',aux=False, se_loss=False)
    model = model.cuda()

    # build optimizer
    optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)

    # load pretrained model if exists
    if args.pretrained_model_path is not None:
        print('load model from %s ...' % args.pretrained_model_path)
        model.module.load_state_dict(torch.load(args.pretrained_model_path))
        print('Done!')
    # train
    train(args, model, optimizer, dataloader_train)

if __name__ == '__main__':
    params = [
        '--num_epochs', '51',
        '--learning_rate', '0.0001',
        '--cuda', '0',
        '--save_model_path', './models',
        '--checkpoint_step','5'
    ]
    main(params)

