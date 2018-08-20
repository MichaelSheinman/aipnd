# Author: Michael Sheinman 
# Started: Tuesday, August 14
# File: predict.py  
# Python file used to predict image from the network 
import argparse

import numpy as np

import json
import torch 
from train import create_model 
from torch import nn, optim
from torch.autograd import Variable 
import torch.nn.functional as F
from torchvision import datasets, transforms, models 
from collections import OrderedDict 

from PIL import Image

    
parser = argparse.ArgumentParser()
    
parser.add_argument('--image', type=str, help='path to the image for testing')
parser.add_argument('--top_k', type=int, help='Top classes to return', default=5)
parser.add_argument('--checkpoint', type=str, help='Checkpoint to use the file with') 
parser.add_argument('--gpu', action='store_true', help='amount of times the model will train')
parser.add_argument('--labels', type=str, help='file for label names', default='cat_to_name.json')

args, _ = parser.parse_known_args()

def loading_checkpoint(checkpoint):
    
    checkpoint_state = torch.load(checkpoint, map_location = lambda storage, loc: storage)

    class_to_idx = checkpoint_state['class_to_idx']

    # Create from a pre-trained model 
    model, optimizer, criterion = create_model(512, class_to_idx)

    # Load checkpoint state into the model 
    model.load_state_dict(checkpoint_state['state_dict'])
    optimizer.load_state_dict(checkpoint_state['optimizer'])

    # Print contents just to confirm if everything looks okay
    print("Loaded a checkpoint => {} with arch {}, hidden units {} and epochs {}".format
          (checkpoint, 
           checkpoint_state['arch'], 
           checkpoint_state['hidden_units'], 
           checkpoint_state['epochs']))
    return model, optimizer, criterion


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    width, height = image.size
    tgt_size = 256
    
    
    height = [int(tgt_size) if height > width else int(max(height * tgt_size/width, 1))][0]
    width = [int(tgt_size) if height < width else int(max(height * tgt_size/width, 1))][0]

    resized_img = image.resize((width, height)) 
    
    tgt_size2 = 224
    width2, height2 = resized_img.size 
    x1 = (width2 - tgt_size2) / 2
    x2 = (height2 - tgt_size2) / 2
    x3 = x1 + tgt_size2
    x4 = x2 + tgt_size2
    
    crop_img = resized_img.crop((x1, x2, x3, x4))
    
    np_image = np.array(crop_img)/255. 
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])     
    np_image = (np_image - mean) / std
    
    # Transpose to map color channel to first dimension 
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image


def predict(image, checkpoint, topk, labels, gpu=False):
    model, optimizer, criterion = loading_checkpoint(checkpoint)
    if gpu and torch.cuda.is_available():
        model.cuda()
    model.eval()
    img = Image.open(image)
    image = process_image(img)
    if gpu:
        myInput = torch.FloatTensor(image).cuda()
    else:
        myInput = torch.FloatTensor(image)
    myInput.unsqueeze_(0)
    output = model(myInput)
    ps = F.softmax(output, dim = 1)
    probs, classes = torch.topk(ps, topk)
    inverted_class_2_index = {model.class_to_idx[x]: x for x in model.class_to_idx}
    new_classes = []
    
    for index in classes.cpu().numpy()[0]:
        new_classes.append(inverted_class_2_index[index])
        
    return probs.cpu().detach().numpy()[0], new_classes
                         
if args.image and args.checkpoint:
    probs, classes = predict(args.image, args.checkpoint, args.top_k, args.labels, args.gpu)
    with open(args.labels, 'r') as f:
        cat_to_name = json.load(f)
    max_index = np.argmax(probs)
    max_prob = probs[max_index]
    max_classes = classes[max_index]
    first = cat_to_name[max_classes]
    print("---------Classes and Probabilities---------")
    for i, index in enumerate(classes):
        print("Class:", cat_to_name[index], "Probability:", probs[i])
    
   
  
