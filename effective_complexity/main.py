import matplotlib.pyplot as plt
from effective_complexity.data import apply_pca, apply_tsne, show_distrib, show_gaussian
from effective_complexity.datasets import collate_fn
import numpy as np
#from effective_complexity.model import MLP, initialize_weights
import torch.nn as nn
import torch
from tqdm import tqdm
import torch.optim as optim
import os
from datetime import datetime

from effective_complexity.functions import train_loop, val_loop, test_loop
from torch.utils.data import DataLoader


def identify(dataloaders, model, hyperparams, train):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #train_dataset, val_dataset, test_dataset = dataset
    train_loader, val_loader, test_loader = dataloaders
    general_hyperparams, model_hyperparams, dataset_hyperparams = hyperparams

    #CREATE SYNTHETIC DATASET
    epochs = general_hyperparams['num_epochs']
    lr = general_hyperparams['learning_rate']
    embedding_size = general_hyperparams['embedding_size']

    model_name = model.__class__.__name__
    dataset_name = train_loader.dataset.__class__.__name__
    folder_name = ''+dataset_name+'_'+model_name+'_'+datetime.now().strftime("%Y-%m-%d %H:%M")

    
    # Initialize training 
    test_model = model
    criterion = nn.KLDivLoss(size_average=None, reduce=None, reduction='batchmean', log_target=False)
    optimizer = optim.Adam(test_model.parameters(), lr=lr)
    best_val_loss = float('inf')

    checkpoints_folder_path = os.path.join(os.getcwd(), 'CHECKPOINTS', folder_name)
    os.makedirs(checkpoints_folder_path, exist_ok=True)

    if not train:
        print('Testing...')
        model_path = os.path.join(checkpoints_folder_path, 'best_model.pth')
        if not os.path.isfile(model_path):
            print(f"No model found at {model_path}.")
            return
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

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(test_model.state_dict(), os.path.join(checkpoints_folder_path, 'best_model.pth'))

    # Final Test Evaluation
    test_loss, test_acc, predictions = test_loop(test_loader, test_model, criterion, device)
    print("\nFinal Test Results")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    

    #PLOT WITH PCA AND TSNE (WHAT DO I PLOT?)
    # Number of principal components to test
    num_components = np.arange(1, embedding_size + 1)
    errors = []


    predicted_distrib = torch.cat(predictions)
    real_distrib = torch.cat([batch['label'] for batch in test_loader])
    # Compute reconstruction error for different numbers of components
    for n in num_components:
        pcareduced_distrib, reconstructed_data = apply_pca(predicted_distrib, num_components=n)
        reconstruction_error = np.mean((predicted_distrib.cpu().numpy() - reconstructed_data) ** 2)
        errors.append(reconstruction_error)



    
    plots_folder_path = os.path.join(os.getcwd(), 'PLOTS', folder_name)
    os.makedirs(plots_folder_path, exist_ok=True)


    # Plot Reconstruction Error vs. Number of Components
    plt.figure(figsize=(8, 5))
    plt.plot(num_components, errors, marker='o', linestyle='-')
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Reconstruction Error (MSE)")
    plt.title("PCA Reconstruction Error vs. Number of Components")
    plt.grid(True)
    plt.savefig(os.path.join(plots_folder_path, 'PCA_analysis.png'))
    #plt.show()

    min_index = errors[:3].index(min(errors[:3]))
    print('num optimal dimensions:', min_index + 1)
    if min_index < 4: #Must be
        num_components = min_index + 1
        pcareduced_pred_distrib, reconstructed_data = apply_pca(predicted_distrib, num_components=num_components)
        tsnereduced_pred_distrib = apply_tsne(predicted_distrib, num_components=num_components)

        pcareduced_distrib, _ = apply_pca(real_distrib, num_components=num_components)
        tsnereduced_distrib = apply_tsne(real_distrib, num_components=num_components)

        #show_distrib(predicted_distrib, predicted=True)
        show_distrib(pcareduced_pred_distrib, plots_folder_path, method='PCA', predicted=True)
        show_distrib(tsnereduced_pred_distrib, plots_folder_path, method='TSNE', predicted=True)
        #show_distrib(real_distrib, predicted=False)
        show_distrib(pcareduced_distrib, plots_folder_path, method='PCA', predicted=False)
        show_distrib(tsnereduced_distrib, plots_folder_path,  method='TSNE', predicted=False)

    


    
        





