# Author: Michael Sheinman 
# Started: Tuesday, August 14
# File: train.py 
# Python file used to train a network 

# imports
import numpy as np

import torch 
from torch import nn, optim
from torch.autograd import Variable 
import torch.nn.functional as F
from torchvision import datasets, transforms, models 
from collections import OrderedDict 
import sys 

from PIL import Image

import argparse

    
def create_model(hidden_units, class_to_index, arch = 'densenet121', learning_rate=0.001, version=1):
    """
    arch: currently supports densenet121(default), vgg19, vgg16 and alexnet. 
    returns: The newly created model 
    """
    # debug 
    if arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        raise TypeError("The arch specified is not supported")
    
    # Freezing parameters so we don't backpropagate through them 
    for param in model.parameters():
        param.requires_grad = False

    output_size = 128  
    # Added due to review so training will be possible for all archs
    input_size = model.classifier.in_features

                       
    # Building the classifier 
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_size, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_units, output_size)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    
    model.classifier = classifier
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    criterion = nn.NLLLoss()
    model.class_to_idx = class_to_index
    
    return model, optimizer, criterion 

def eval_model(testloader, model, criterion, device, is_gpu):
    running_loss = 0 
    accuracy = 0
    
    model.eval()
    if is_gpu:
        model.cuda()
   
    for ii, (inputs, labels) in enumerate(testloader): 
        inputs, labels = inputs.to(device), labels.to(device)
        
        if ii % 3 == 0:
            sys.stdout.write('.')
        output = model.forward(inputs)
        running_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(1)[1])
        # I usetype_as(torch.FloatTensor()) based on a code 
        # Caleb Josue Ruiz Torres posted in the slank channel 
        # long time ago and I found to fix mismatch error. 
        accuracy += equality.type_as(torch.FloatTensor()).mean()  
        
    return (running_loss/len(testloader)),(accuracy/len(testloader))


def save_checkpoint(arch, state_dic, op_dict, lr, hidden, epochs, c_idx):
    return {
        'arch': arch,
        'state_dict': state_dic,
        'optimizer': op_dict,
        'learning_rate': lr,
        'hidden_units': hidden,
        'epochs': epochs,
        'class_to_idx': c_idx
    }


def train_model(image_datasets, dataloaders,  arch, hidden_units, epochs, learning_rate, gpu=False, checkpoint=''):
    dataset_sizes = {
        x: len(dataloaders[x].dataset) 
        for x in list(image_datasets.keys())
    }    
    
    
    # Create the model 
    train_data = dataloaders['training']
    validation_data = dataloaders['validation']
    my_data = image_datasets['training']
    num_labels = len(my_data.classes)
    class_to_index = my_data.class_to_idx
    model, optimizer, criterion = create_model(class_to_index=class_to_index, hidden_units=hidden_units, arch=arch ,learning_rate=learning_rate)
    # debug info
    print(gpu)
    # print("The checkpooint is: ", checkpoint)
    if gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        model.cuda()
        # debug info:
        # print("Enabling gpu the device is:", device)
    else:
        device = torch.device("cpu")
    
    # Train the model 
    sys.stdout.write("Training")
    steps = 0
    print_every = 40
    for e in range(epochs):
        running_loss = 0 
        for ii, (inputs, labels) in enumerate(train_data):
            model.train()
            steps += 1 
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step() 
            if ii % 4 == 0:
                sys.stdout.write('.')
                sys.stdout.flush()
            running_loss +=  loss.item()
            

            if steps % print_every == 0:
                model.eval()
                eval_loss, eval_accuracy = eval_model(validation_data, model, criterion, device, gpu)
                print("Epoch: {}/{}".format(e+1, epochs), "Loss: {:.3f}".format(running_loss/print_every), 
                     "Validation Loss: {:.3f}".format(eval_loss), "Validation accuracy: {:.3f}".format(eval_accuracy))
                print()
                sys.stdout.write("Training")
                running_loss = 0 
    print("Training is now complete!")

    if checkpoint:
        checkpoint_saved = save_checkpoint('densenet121', model.state_dict(), optimizer.state_dict(), learning_rate, hidden_units, 
                                           epochs, model.class_to_idx)
        torch.save(checkpoint_saved, checkpoint)
    return model         


    
def main():
    print(args.data_dir)
    if args.data_dir:
        print("here")
        arch = args.arch
        learning_rate = args.learning_rate
        hidden_units = args.hidden_units
        epochs = args.epochs
        
        train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
        
        validation_transforms =  transforms.Compose([transforms.Resize(256),   
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),    
                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])
        testing_transforms =  transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
        data_dir = 'flowers'
        train_dir = data_dir + '/train'
        valid_dir = data_dir + '/valid'
        test_dir = data_dir + '/test'
        image_datasets = {
            'training': datasets.ImageFolder(train_dir, transform=train_transforms),
            'validation': datasets.ImageFolder(valid_dir, transform=validation_transforms),
            'testing': datasets.ImageFolder(test_dir, transform=testing_transforms)
        }
        dataloaders = {
            'training': torch.utils.data.DataLoader(image_datasets['training'], batch_size = 64, shuffle=True),
            'validation': torch.utils.data.DataLoader(image_datasets['validation'], batch_size = 64, shuffle=True),
            'testing': torch.utils.data.DataLoader(image_datasets['testing'], batch_size = 32, shuffle=False)
        }
        checkpoint = ''
        if args.save_dir:
            checkpoint = args.save_dir
        if args.gpu:
            gpu = True
        else:
            gpu = False 
        
        train_model(image_datasets, dataloaders, arch=args.arch, hidden_units = args.hidden_units, 
                   epochs=args.epochs, learning_rate=args.learning_rate, checkpoint=checkpoint, gpu=gpu) 

        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, help='path to folder of images')
    parser.add_argument('--arch', type=str, default='densenet121', 
                            help='chosen model')
    parser.add_argument('--learning_rate', type=float, default=0.01, help="learn rate")
    parser.add_argument('--hidden_units', type=int, default=512, 
                            help='amount of hidden units')
    parser.add_argument('--epochs', type=int, default=1,
                            help='amount of times the model will train')
    parser.add_argument('--gpu', action='store_true', help='amount of times the model will train')
    parser.add_argument('--save_dir', type=str, help='determine whether to save checkpoint')
    args, _ = parser.parse_known_args()

    main()

