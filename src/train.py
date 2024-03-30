import torch
from tqdm import tqdm
from util import *

def train_eval(train_loader, val_loader, model, criterion, optimizer, num_epochs, device, checkpoint_path='../checkpoints/model_weights.pth'):

    # Train
    for epoch in range(num_epochs):
        train_correct = 0
        for batch_idx_train, (X_train, y_train) in enumerate(train_loader):
            X = X_train.to(device)
            y = y_train.to(device)
            
            y_pred = model(X)
            train_loss = criterion(y_pred,y)

            y_pred_labels = torch.argmax(y_pred.data, dim=1)
            train_correct += (y_pred_labels == y).sum().item()

            optimizer.zero_grad()
            train_loss.backward()
            
            optimizer.step()

        avg_train_acc = train_correct / len(train_loader.dataset)
        print(f'Epochs:{epoch+1}, Train Loss:{train_loss.data:0.4f}, Train Accuracy :{avg_train_acc}')
        torch.save(model.state_dict(), checkpoint_path)


        val_loss = 0
        val_correct = 0
        print()
        print('Computing validation performace...')
        with torch.no_grad():
          for batch_idx_val, (X_val, y_val) in enumerate(val_loader):
              X = X_val.to(device=device)
              y = y_val.to(device=device)

              y_pred = model(X)
              val_loss += criterion(y_pred,y).data

              y_pred_labels = torch.argmax(y_pred, dim=1)
              val_correct += (y_pred_labels == y).sum().item()

        avg_val_acc = val_correct / len(val_loader.dataset)
        avg_val_loss = val_loss / len(val_loader)

        print(f'Average Val Loss:{avg_val_loss:0.4f}, Average Val Accuracy : {avg_val_acc:0.2f}')
        print()



