from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
parser.add_argument('--image_folder', type=str, default='data/samples', help='path to dataset')
parser.add_argument('--batch_size', type=int, default=50, help='size of each image batch')
parser.add_argument('--model_config_path', type=str, default='config/v0904yolov3.cfg', help='path to model config file')
parser.add_argument('--data_config_path', type=str, default='config/v0904.data', help='path to data config file')
parser.add_argument('--weights_path', type=str, default='checkpoints/yolov3.weights', help='path to weights file')
parser.add_argument('--class_path', type=str, default='data/v0904.names', help='path to class label file')
parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=128, help='size of each image dimension')
parser.add_argument('--max_objects', type=int, default=5, help='maximum number of objects to detect')
parser.add_argument('--checkpoint_interval', type=int, default=1, help='interval between saving model weights')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='directory where model checkpoints are saved')
parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
opt = parser.parse_args()
#print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda

os.makedirs('output', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

classes = load_classes(opt.class_path)

# Get data configuration
data_config     = parse_data_config(opt.data_config_path)
train_path      = data_config['train']

# Get hyper parameters
hyperparams     = parse_model_config(opt.model_config_path)[0]
learning_rate   = float(hyperparams['learning_rate'])
momentum        = float(hyperparams['momentum'])
decay           = float(hyperparams['decay'])
burn_in         = int(hyperparams['burn_in'])

# Initiate model
model = Darknet(opt.model_config_path,opt.img_size)

#try:
    #model.load_weights(opt.weights_path)
#except:
    #print("Weight Loading Failed.")

model.apply(weights_init_normal)

if cuda:
    model = model.cuda()

model.train()

# Get dataloader
dataloader = torch.utils.data.DataLoader(
    ListDataset(train_path,opt.img_size,opt.max_objects),
    batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

optimizer = optim.Adam(model.parameters(), lr=1e-4)

print_per_batch = False

for epoch in range(opt.epochs):

    losses = 0
    counter = 0

    for batch_i, (_, imgs, targets) in enumerate(tqdm(dataloader,total = len(dataloader))):
        imgs = Variable(imgs.type(Tensor))
        targets = Variable(targets.type(Tensor), requires_grad=False)

        optimizer.zero_grad()

        loss = model(imgs, targets)

        loss.backward()
        optimizer.step()

        losses += np.asarray([model.losses['x'], model.losses['y'],model.losses['w'],model.losses['h'],model.losses['conf'],model.losses['cls'],loss.item(),model.losses['recall']])
        counter += 1
        
        if print_per_batch:
            print('[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f]' %
                                        (epoch+1, opt.epochs, batch_i, len(dataloader),
                                        model.losses['x'], model.losses['y'], model.losses['w'],
                                        model.losses['h'], model.losses['conf'], model.losses['cls'],
                                        loss.item(), model.losses['recall']))
        
        model.seen += imgs.size(0)
        
    # Get Statistics
    losses /= counter
    print('[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f]' %
                                        (epoch+1, opt.epochs, len(dataloader), len(dataloader), *losses))

    if epoch % opt.checkpoint_interval == 0:
        model.save_weights('%s/yolov3.weights' % (opt.checkpoint_dir))
