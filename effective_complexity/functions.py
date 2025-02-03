import torch
import torch.nn as nn
import numpy as np
from effective_complexity.data import sample_gaussian_3d
from effective_complexity.model import MLP
from tqdm import tqdm
from effective_complexity.data import SYNTHETIC_DATA
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

    #MANUAL SOFTMAX
    #logits = [torch.exp(torch.matmul(f_x.float(), (torch.matmul(W, one_hots[i]).t()))) for i in range(3)]
    #print('logits ', logits)
    #logits = torch.stack(logits)
    #denom = torch.sum(logits)
    #label = [torch.div(logit,denom) for logit in logits]

    assert label[0]+label[1]+label[2] < 1 + eps and label[0]+label[1]+label[2] > 1 - eps
    
    return label

def initialize_synthetic_exp(DIMS,MU,COV, num_samples):
    mu_sample = np.full(DIMS,MU)
    covariance_sample = np.diag(np.full(DIMS,COV))

    # Sample points
    samples = sample_gaussian_3d(mu_sample, covariance_sample, num_samples)
    samples = torch.from_numpy(samples).float()
    #Define orthogonal vectors
    w1 = torch.tensor([1,2,2])
    w2 = torch.tensor([-2,1,0])
    w3 = torch.tensor([-2,-2,1])
    W = torch.stack([w1,w2,w3], dim = 0).t().float()
    return W,samples

def compute_fx(samples, input_size, hidden_sizes, output_size):
    mlp_model = MLP(input_size, hidden_sizes, output_size)
    #mlp_model.apply(initialize_weights)
    f_x = torch.stack([mlp_model(x) for x in samples], dim=0)
    return f_x


def generate_labels(f_x, W, num_samples):
    train_dataset = SYNTHETIC_DATA()
    test_dataset = SYNTHETIC_DATA()
    val_dataset = SYNTHETIC_DATA()

    train_percent=0.7
    test_percent = 0.2
    val_percent = 0.1

    print('Computing ground truth labels')
    for x in tqdm(f_x):
        if len(train_dataset) < math.floor(num_samples*train_percent):
            train_dataset.add_item({'x':x, 'label':compute_distrib(x,W)})
        elif len(val_dataset) < math.floor(num_samples*val_percent):
            val_dataset.add_item({'x':x, 'label':compute_distrib(x,W)})
        else:
            test_dataset.add_item({'x':x, 'label':compute_distrib(x,W)})

    return train_dataset, val_dataset, test_dataset



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
        log_outputs = logsoftmax(outputs)
        #MINIMIZE KL DIVERGENCE
        loss = criterion(log_outputs, labels)
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
            log_outputs = logsoftmax(outputs)
            loss = criterion(log_outputs, labels)
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
            log_outputs = logsoftmax(outputs)
            loss = criterion(log_outputs, labels)
            loss += elastic_net_regularization(model, l1_lambda, l2_lambda, l1_weight, l2_weight)


            total_loss += loss.item()
            #Accuracy computation
            #_, predicted = torch.max(outputs, 1)
            #correct += (predicted == labels).sum().item()
            #total += labels.size(0)

    avg_loss = total_loss / len(test_loader)
    #accuracy = correct / total
    return avg_loss, accuracy, total_outputs
            