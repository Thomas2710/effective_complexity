import matplotlib.pyplot as plt
from effective_complexity.data import collate_fn, apply_pca, apply_tsne, show_distrib, show_gaussian
import numpy as np
from effective_complexity.model import MLP, initialize_weights
import torch.nn as nn
import torch
from tqdm import tqdm
import torch.optim as optim

from effective_complexity.functions import train_loop, val_loop, test_loop, initialize_synthetic_exp, compute_fx, generate_labels

from torch.utils.data import DataLoader


def identify(model, hyperparams):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #CREATE SYNTHETIC DATASET
    DIMS = hyperparams['DIMS']
    COV = hyperparams['COV']
    MU = hyperparams['MU']
    BATCH_SIZE = hyperparams['BATCH_SIZE']
    epochs = hyperparams['num_epochs']
    lr = hyperparams['learning_rate']
    # Number of points to sample
    num_samples = hyperparams['num_samples']



    # Instantiate the MLP (f(x))
    input_size = 3  # Number of input features
    hidden_sizes = [64, 32, 64]  # Hidden layer sizes
    output_size = 3  # Number of output features (e.g., classification classes)

    #Create gaussian distributed samples and orthogonal vectors in matrix W
    W, samples = initialize_synthetic_exp(DIMS, MU, COV, num_samples)

    #Compute label distribution
    with torch.no_grad():
        f_x = compute_fx(samples, input_size, hidden_sizes, output_size)
        #Get distribution label
        train_dataset, val_dataset, test_dataset = generate_labels(f_x, W, num_samples)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size = 128, shuffle = True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True, collate_fn=collate_fn)


    #INSTANTIATE MODEL
    #TRAIN MODEL MINIMIZING KL DIVERGENCE 
    #test_model = MLP(input_size, hidden_sizes, output_size)
    test_model = model
    criterion = nn.KLDivLoss(size_average=None, reduce=None, reduction='batchmean', log_target=False)
    optimizer = optim.Adam(test_model.parameters(), lr=lr)
 
    test_model.to(device)

    # Print epoch loss
    print('Training ...')
    progress_bar = tqdm(range(epochs))
    for epoch in progress_bar:
        train_loss, train_acc = train_loop(train_loader,test_model, criterion, optimizer, device)
        val_loss, val_acc = val_loop(val_loader, test_model, criterion, device)

        progress_bar.set_postfix(Epoch = f"[{epoch+1}/{epochs}]")
        progress_bar.set_postfix(Train = f"Loss: {train_loss:.4f}, Acc: {train_acc:.4f}" )
        progress_bar.set_postfix(Val = f"Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    # Final Test Evaluation
    test_loss, test_acc, predicted_distrib = test_loop(test_loader, test_model, criterion, device)
    print("\nFinal Test Results")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    


    #PLOT WITH PCA AND TSNE (WHAT DO I PLOT?)

    predicted_distrib = torch.cat(predicted_distrib)
    pcareduced_pred_distrib = apply_pca(predicted_distrib)
    tsnereduced_pred_distrib = apply_tsne(predicted_distrib)


    real_distrib = torch.stack([item['label'] for item in test_dataset])
    pcareduced_distrib = apply_pca(real_distrib)
    tsnereduced_distrib = apply_tsne(real_distrib)

    show_distrib(predicted_distrib, predicted=True)
    show_distrib(pcareduced_pred_distrib, method='PCA', predicted=True)
    show_distrib(tsnereduced_pred_distrib, method='TSNE', predicted=True)
    show_distrib(real_distrib, predicted=False)
    show_distrib(pcareduced_distrib, method='PCA', predicted=False)
    show_distrib(tsnereduced_distrib, method='TSNE', predicted=False)

    


    
        





