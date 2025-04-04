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

#Elastic regularization, allows to customize the weight of L1 and L2 regularization
def elastic_net_regularization(model, l1_lambda, l2_lambda, l1_weight, l2_weight):
    l1_penalty = sum(p.abs().sum() for p in model.parameters())  # L1 norm
    l2_penalty = sum((p ** 2).sum() for p in model.parameters())  # L2 norm
    return l1_weight * l1_lambda * l1_penalty + l2_weight * l2_lambda * l2_penalty

#Train loop
def train_loop(train_loader, model, criterion, optimizer, device = 'cpu'):
    model.train()
    logsoftmax = nn.LogSoftmax(dim=-1)
    total_loss = []
    accuracy = 0
    total = 0
    correct = 0

    d = model.get_W().shape[0]
    one_hots = torch.eye(d).float()


    for batch in train_loader:
        optimizer.zero_grad()
        labels = batch['label'].to(device)
        inputs = batch['x'].to(device)
        embedding_output = model(inputs)
        #Compute f_x from output
        f_x = model.get_fx(embedding_output)
        #Compute unenmbed from output
        unembedding = model.get_unembeddings(one_hots)
        logits = torch.matmul(f_x, unembedding)

        outputs = logsoftmax(logits)
        loss = criterion(outputs, labels)
        #loss += elastic_net_regularization(model, l1_lambda, l2_lambda, l1_weight, l2_weight)

        total_loss.append(loss.item())
        '''
        #Accuracy computation
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)'
        '''

        loss.backward()
        optimizer.step()

    avg_loss = sum(total_loss) / len(total_loss)
    return avg_loss, accuracy


#Validation loop
def val_loop(val_loader, model, criterion, device = 'cpu'):
    model.eval()
    total_loss = 0.0
    accuracy = 0
    total = 0
    correct = 0
    logsoftmax = nn.LogSoftmax(dim=-1)

    d = model.get_W().shape[0]
    one_hots = torch.eye(d).float()


    with torch.no_grad(): 
        for batch in val_loader:
            inputs = batch['x'].to(device)
            labels = batch['label'].to(device)
            outputs = model(inputs)

            #Compute f_x from output
            f_x = model.get_fx(outputs)
            #Compute unenmbed from output
            unembedding = model.get_unembeddings(one_hots)

            logits = torch.matmul(f_x, unembedding)
            outputs = logsoftmax(logits)
            loss = criterion(outputs, labels)
            #loss += elastic_net_regularization(model, l1_lambda, l2_lambda, l1_weight, l2_weight)


            total_loss += loss.item()
            '''
            #Accuracy computation
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)'
            '''

    avg_loss = total_loss / len(val_loader)
    return avg_loss, accuracy

#Test loop
def test_loop(test_loader, model, criterion, device='cpu'):
    model.eval()  
    total_loss = 0.0
    accuracy = 0
    total = 0
    correct = 0
    logsoftmax = nn.LogSoftmax(dim=-1)
    softmax = nn.Softmax(dim=-1)
    total_fx = []
    outputs_to_return = []

    d = model.get_W().shape[0]
    one_hots = torch.eye(d).float()


    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['x'].to(device)
            labels = batch['label'].to(device)
            outputs = model(inputs)

            #Compute f_x from output
            f_x = model.get_fx(outputs)
            total_fx.append(f_x)
            #Compute unembeddings
            unembedding = model.get_unembeddings(one_hots)

            logits = torch.matmul(f_x, unembedding)
            outputs = logsoftmax(logits)
            outputs_to_return.append(softmax(logits))

            loss = criterion(outputs, labels)
            #loss += elastic_net_regularization(model, l1_lambda, l2_lambda, l1_weight, l2_weight)
            total_loss += loss.item()
            '''
            #Accuracy computation
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            '''

    avg_loss = total_loss / len(test_loader)
    return avg_loss, accuracy, total_fx, outputs_to_return
