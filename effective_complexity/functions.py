import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import math


eps = 1e-6
l1_lambda = 1e-4
l2_lambda = 1e-4
l1_weight = 0
l2_weight = 0

def compute_distrib(f_x, W, log = False):
    one_hots = torch.eye(W.shape[0]).float()
    logits = [torch.matmul(f_x.float(), (torch.matmul(W, one_hots[i]).t())) for i in range(3)]
    logits = torch.stack(logits)

    if log:
        softmax = nn.LogSoftmax(dim=0)
    else:
        softmax = nn.Softmax(dim=0)

    label = softmax(logits)
    assert label[0]+label[1]+label[2] < 1 + eps and label[0]+label[1]+label[2] > 1 - eps
    
    return label

def elastic_net_regularization(model, l1_lambda, l2_lambda, l1_weight, l2_weight):
    l1_penalty = sum(p.abs().sum() for p in model.parameters())  # L1 norm
    l2_penalty = sum((p ** 2).sum() for p in model.parameters())  # L2 norm
    return l1_weight * l1_lambda * l1_penalty + l2_weight * l2_lambda * l2_penalty



def train_loop(train_loader, model, criterion, optimizer, device = 'cpu'):
    model.train()
    logsoftmax = nn.LogSoftmax(dim=-1)
    total_loss = 0
    accuracy = 0
    total = 0
    correct = 0

    for batch in train_loader:
        optimizer.zero_grad()
        labels = batch['label'].to(device)
        inputs = batch['x'].to(device)
        outputs = model(inputs)
        outputs = logsoftmax(outputs)
        #MINIMIZE KL DIVERGENCE
        loss = criterion(outputs, labels)
        loss += elastic_net_regularization(model, l1_lambda, l2_lambda, l1_weight, l2_weight)

        total_loss += loss.item()
        #Accuracy computation
        #_, predicted = torch.max(outputs, 1)
        #correct += (predicted == labels).sum().item()
        #total += labels.size(0)

        # Backward pass
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    #accuracy = correct / total
    return avg_loss, accuracy



def val_loop(val_loader, model, criterion, device = 'cpu'):
    model.eval()  # Set model to evaluation mode (no gradient computation)
    total_loss = 0.0
    accuracy = 0
    total = 0
    correct = 0
    logsoftmax = nn.LogSoftmax(dim=-1)


    with torch.no_grad():  # No need to track gradients
        for batch in val_loader:
            inputs = batch['x'].to(device)
            labels = batch['label'].to(device)

            outputs = model(inputs)
            outputs = logsoftmax(outputs)
            loss = criterion(outputs, labels)
            loss += elastic_net_regularization(model, l1_lambda, l2_lambda, l1_weight, l2_weight)


            total_loss += loss.item()
            #Accuracy computation
            #_, predicted = torch.max(outputs, 1)
            #correct += (predicted == labels).sum().item()
            #total += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    #accuracy = correct / total
    return avg_loss, accuracy


def test_loop(test_loader, model, criterion, device='cpu'):
    model.eval()  # Set model to evaluation mode (no gradient computation)
    total_loss = 0.0
    accuracy = 0
    total = 0
    correct = 0
    logsoftmax = nn.LogSoftmax(dim=-1)
    total_outputs = []


    with torch.no_grad():  # No need to track gradients
        for batch in test_loader:
            inputs = batch['x'].to(device)
            labels = batch['label'].to(device)

            outputs = model(inputs)
            total_outputs.append(outputs)
            outputs = logsoftmax(outputs)
            loss = criterion(outputs, labels)
            loss += elastic_net_regularization(model, l1_lambda, l2_lambda, l1_weight, l2_weight)


            total_loss += loss.item()
            #Accuracy computation
            #_, predicted = torch.max(outputs, 1)
            #correct += (predicted == labels).sum().item()
            #total += labels.size(0)

    avg_loss = total_loss / len(test_loader)
    #accuracy = correct / total
    return avg_loss, accuracy, total_outputs
            