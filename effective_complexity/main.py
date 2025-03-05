import matplotlib.pyplot as plt
from effective_complexity.data import apply_pca, apply_tsne, show_distrib, show_gaussian
from effective_complexity.datasets import collate_fn
import numpy as np
#from effective_complexity.model import MLP, initialize_weights
import torch.nn as nn
import torch
from tqdm import tqdm
import torch.optim as optim

from effective_complexity.functions import train_loop, val_loop, test_loop
from torch.utils.data import DataLoader


def identify(dataloaders, model, hyperparams):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #train_dataset, val_dataset, test_dataset = dataset
    train_loader, val_loader, test_loader = dataloaders
    general_hyperparams, model_hyperparams, dataset_hyperparams = hyperparams

    #CREATE SYNTHETIC DATASET
    epochs = general_hyperparams['num_epochs']
    lr = general_hyperparams['learning_rate']
    embedding_size = general_hyperparams['embedding_size']
    

    #LOAD DATALOADERS
    #train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    #val_loader = DataLoader(val_dataset, batch_size = 128, shuffle = True, collate_fn=collate_fn)
    #test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True, collate_fn=collate_fn)


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

    # Plot Reconstruction Error vs. Number of Components
    plt.figure(figsize=(8, 5))
    plt.plot(num_components, errors, marker='o', linestyle='-')
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Reconstruction Error (MSE)")
    plt.title("PCA Reconstruction Error vs. Number of Components")
    plt.grid(True)
    plt.show()

    min_index = errors[:3].index(min(errors[:3]))
    print('num optimal dimensions:', min_index + 1)
    if min_index < 4: #Must be
        num_components = min_index + 1
        pcareduced_pred_distrib, reconstructed_data = apply_pca(predicted_distrib, num_components=num_components)
        tsnereduced_pred_distrib = apply_tsne(predicted_distrib, num_components=num_components)


        
        pcareduced_distrib, _ = apply_pca(real_distrib, num_components=num_components)
        tsnereduced_distrib = apply_tsne(real_distrib, num_components=num_components)

        model_name = model.__class__.__name__
        dataset_name = train_loader.dataset.__class__.__name__
        #show_distrib(predicted_distrib, predicted=True)
        show_distrib(pcareduced_pred_distrib, method='PCA', predicted=True, experiment = (dataset_name, model_name))
        show_distrib(tsnereduced_pred_distrib, method='TSNE', predicted=True, experiment = (dataset_name, model_name))
        #show_distrib(real_distrib, predicted=False)
        show_distrib(pcareduced_distrib, method='PCA', predicted=False, experiment = (dataset_name, model_name))
        show_distrib(tsnereduced_distrib, method='TSNE', predicted=False, experiment = (dataset_name, model_name))

    


    
        





