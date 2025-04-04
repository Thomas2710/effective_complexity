import matplotlib.pyplot as plt
from effective_complexity.data import apply_pca, apply_tsne, show_distrib, show_gaussian, find_distribution_limits
from effective_complexity.datasets import collate_fn
import numpy as np
import torch.nn as nn
import torch
from tqdm import tqdm
import torch.optim as optim
import os
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

from effective_complexity.functions import train_loop, val_loop, test_loop
from torch.utils.data import DataLoader





def identify(dataloaders, model, hyperparams, train):

    #------------------------------
    # Problem initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = dataloaders
    general_hyperparams, model_hyperparams, dataset_hyperparams = hyperparams

    epochs = general_hyperparams['num_epochs']
    lr = general_hyperparams['learning_rate']
    embedding_size = general_hyperparams['embedding_size']
    num_classes = general_hyperparams['num_classes']

    model_name = model.__class__.__name__
    dataset_name = train_loader.dataset.__class__.__name__
    folder_name = ''+dataset_name+'_'+model_name+'_'+datetime.now().strftime("%Y-%m-%d %H:%M")
    scaler = MinMaxScaler()

    #------------------------------
    # Training initialization
    test_model = model
    criterion = nn.KLDivLoss(size_average=None, reduce=None, reduction='batchmean', log_target=False)
    optimizer = optim.Adam(test_model.parameters(), lr=lr)
    best_val_loss = float('inf')

    checkpoints_folder_path = os.path.join(os.getcwd(), 'CHECKPOINTS', folder_name)
    os.makedirs(checkpoints_folder_path, exist_ok=True)

    #------------------------------
    # Load model if not training
    if not train:
        print('Testing...')
        model_path = os.path.join(checkpoints_folder_path, 'best_model.pth')
        if not os.path.isfile(model_path):
            print(f"No model found at {model_path}.")
            return
        test_model.to(device)
    #------------------------------
    # Train/Test Loop
    else:
        print('Training ...')
        val_loss_over_epochs = []
        train_loss_over_epochs = []
        progress_bar = tqdm(range(epochs))
        for epoch in progress_bar:
            train_loss, train_acc = train_loop(train_loader,test_model, criterion, optimizer, device)
            val_loss, val_acc = val_loop(val_loader, test_model, criterion, device)

            val_loss_over_epochs.append(val_loss)
            train_loss_over_epochs.append(train_loss)
            progress_bar.set_postfix(Epoch = f"[{epoch+1}/{epochs}]")
            progress_bar.set_postfix(Train = f"Loss: {train_loss:.4f}, Acc: {train_acc:.4f}" )
            progress_bar.set_postfix(Val = f"Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(test_model.state_dict(), os.path.join(checkpoints_folder_path, 'best_model.pth'))

    # Final Test Evaluation
    test_loss, test_acc, embeddings, predicted_distrib = test_loop(test_loader, test_model, criterion, device)
    print("\nFinal Test Results")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    

    #------------------------------
    # Merge embeddings for each batch
    embeddings = torch.cat(embeddings)
    predicted_distrib = torch.cat(predicted_distrib)


    #------------------------------
    # Create folder to save plots
    plots_folder_path = os.path.join(os.getcwd(), 'PLOTS', folder_name)
    os.makedirs(plots_folder_path, exist_ok=True)


    #-----------------------------
    # Plot Train and Validation Loss over epochs
    plt.plot(range(1, len(train_loss_over_epochs)+1),train_loss_over_epochs,  label='Train Loss')
    plt.plot(range(1, len(val_loss_over_epochs)+1),val_loss_over_epochs, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_folder_path, 'Loss_over_epochs.png'))
    plt.close()


    #-----------------------------
    # Plot Reconstruction Error vs. Number of Components
    num_components = np.arange(1, embedding_size + 1)
    errors = []
    variance = []

    # Compute reconstruction error for different numbers of components
    for n in num_components:
        pcareduced_distrib, reconstructed_data, variance_explained = apply_pca(embeddings, num_components=n)
        reconstruction_error = np.mean((embeddings.cpu().numpy() - reconstructed_data) ** 2)
        errors.append(reconstruction_error)
        variance.append(variance_explained[-1])


    plt.figure(figsize=(8, 5))
    plt.plot(num_components, errors, color='red', marker='o', linestyle='-')
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Reconstruction Error (MSE)")
    plt.title("PCA Reconstruction Error vs. Number of Components")
    plt.grid(True)
    plt.savefig(os.path.join(plots_folder_path, 'PCA_MSE_analysis.png'))
    plt.close()

    #-----------------------------
    #Plot Variance explained vs. Number Of Components
    plt.figure(figsize=(8, 5))
    plt.plot(num_components, variance, color='green', marker='o', linestyle='-')
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Variance Explained")
    plt.title("PCA Variance Explained vs. Number of Components")
    plt.grid(True)
    plt.savefig(os.path.join(plots_folder_path, 'PCA_VAR_analysis.png'))
    plt.close()

    #-----------------------------
    # Plot Embeddings in 2D and 3D
    reference_embeddings = test_loader.dataset.f_x

    embeddings = scaler.fit_transform(embeddings)
    reference_embeddings = scaler.fit_transform(reference_embeddings)

    min_index = errors[:3].index(min(errors[:3]))
    print('num optimal dimensions for embedding:', min_index + 1)
    if min_index < 4 and min_index > 1:
        num_components = min_index + 1
        projection = '3d' if num_components == 3 else None
        if embeddings.shape[1] > 3:
            #PCA
            pcareduced_pred_embeddings, reconstructed_data, _ = apply_pca(embeddings, num_components=num_components)
            pcareduced_embeddings, _ , _ = apply_pca(reference_embeddings, num_components=num_components)
            pca_fig, pca_axs = plt.subplots(1, 2, figsize=(12, 10),
                            subplot_kw={'projection': projection} if projection else {})
            limit = find_distribution_limits(pcareduced_embeddings, pcareduced_pred_embeddings)
            show_distrib(pcareduced_embeddings, method='PCA', predicted=False, ax=pca_axs[0], limit = limit)
            show_distrib(pcareduced_pred_embeddings, method='PCA', predicted=True, ax=pca_axs[1], limit = limit)
            pca_fig.tight_layout()
            plt.savefig(os.path.join(plots_folder_path,''+str(pcareduced_embeddings.shape[1])+'dim_embeddings_pca.png'))
            plt.close()

            #TSNE
            tsnereduced_pred_embeddings = apply_tsne(embeddings, num_components=num_components)
            tsnereduced_embeddings = apply_tsne(reference_embeddings, num_components=num_components)
            tsne_fig, tsne_axs = plt.subplots(1, 2, figsize=(12, 10),
                        subplot_kw={'projection': projection} if projection else {})
            limit = find_distribution_limits(tsnereduced_embeddings, tsnereduced_pred_embeddings)
            show_distrib(tsnereduced_embeddings, method='TSNE', predicted=False, ax=tsne_axs[0], limit = limit)
            show_distrib(tsnereduced_pred_embeddings, method='TSNE', predicted=True, ax=tsne_axs[1], limit = limit)
            tsne_fig.tight_layout()
            plt.savefig(os.path.join(plots_folder_path,''+str(tsnereduced_embeddings.shape[1])+'dim_embeddings_tsne.png'))
            plt.close()
        else:
            pred_embeddings = embeddings
            reference_embeddings = reference_embeddings.cpu().detach().numpy()
            tsne_fig, tsne_axs = plt.subplots(1, 2, figsize=(12, 10),
                        subplot_kw={'projection': projection} if projection else {})
            limit = find_distribution_limits(tsnereduced_embeddings, tsnereduced_pred_embeddings)
            show_distrib(reference_embeddings, predicted=False, ax=tsne_axs[0], limit = limit)
            show_distrib(pred_embeddings, predicted=True, ax=tsne_axs[1], limit = limit)
            tsne_fig.tight_layout()
            plt.savefig(os.path.join(plots_folder_path,''+str(reference_embeddings.shape[1])+'dim_embeddings.png'))
            plt.close()



    #-----------------------------
    # Plot PCA and t-SNE distributions
    real_distrib = torch.cat([batch['label'] for batch in test_loader])
    projection = '3d' if num_classes == 3 else None

    real_distrib = scaler.fit_transform(real_distrib)
    predicted_distrib = scaler.fit_transform(predicted_distrib)
    if num_classes > 3:
        #PCA
        pcareduced_pred_distrib, reconstructed_data, _ = apply_pca(predicted_distrib, num_components=num_components)
        pcareduced_distrib, _, _ = apply_pca(real_distrib, num_components=num_components)
        pca_fig, pca_axs = plt.subplots(1, 2, figsize=(12, 10),
                            subplot_kw={'projection': projection} if projection else {})
        limit = find_distribution_limits(pcareduced_distrib, pcareduced_pred_distrib)
        show_distrib(pcareduced_distrib, method='PCA', predicted=False, ax=pca_axs[0], limit = limit)
        show_distrib(pcareduced_pred_distrib, method='PCA', predicted=True, ax=pca_axs[1], limit = limit)
        pca_fig.tight_layout()
        plt.savefig(os.path.join(plots_folder_path,''+str(pcareduced_distrib.shape[1])+'dim_distribution_pca.png'))
        plt.close()

        #TSNE
        tsnereduced_pred_distrib = apply_tsne(predicted_distrib, num_components=num_components)
        tsnereduced_distrib = apply_tsne(real_distrib, num_components=num_components)
        tsne_fig, tsne_axs = plt.subplots(1, 2, figsize=(12, 10),
                    subplot_kw={'projection': projection} if projection else {})
        limit = find_distribution_limits(tsnereduced_distrib, tsnereduced_pred_distrib)
        show_distrib(tsnereduced_distrib, method='TSNE', predicted=False, ax=tsne_axs[0], limit = limit)
        show_distrib(tsnereduced_pred_distrib, method='TSNE', predicted=True, ax=tsne_axs[1], limit = limit)
        tsne_fig.tight_layout()
        plt.savefig(os.path.join(plots_folder_path,''+str(tsnereduced_distrib.shape[1])+'dim_distribution_tsne.png'))
        plt.close()
    else:
        fig, axs = plt.subplots(1, 2, figsize=(12, 10),
                            subplot_kw={'projection': projection} if projection else {})
        real_distrib = real_distrib
        predicted_distrib = predicted_distrib
        limit = find_distribution_limits(real_distrib, predicted_distrib)
        show_distrib(real_distrib, predicted=False, ax=axs[0], limit = limit)
        show_distrib(predicted_distrib, predicted=True, ax=axs[1], limit = limit)
        fig.tight_layout()
        plt.savefig(os.path.join(plots_folder_path,''+str(real_distrib.shape[1])+'dim_distribution.png'))
        plt.close()




    
        





