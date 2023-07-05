import torchvision.models as models
import torch
import time
import matplotlib.pyplot as plt
from torchvision import transforms
from torcheval.metrics.functional import multiclass_f1_score

from tqdm.auto import tqdm
from typing import Dict, List, Tuple



def load_efficient_net():
    model = models.efficientnet_b0(weights="DEFAULT")
    # if pretrain:
    #     weights = models.EfficientNet_B0_Weights.DEFAULT
    for param in model.features.parameters():
        param.requires_grad = False
    model.eval()

    return model


def load_mobile_net():
    model = models.mobilenet_v2(weights="DEFAULT")

    for param in model.features.parameters():
        param.requires_grad = False
    model.eval()

    return model

def preprocess():
    
    transform = transforms.Compose([
                    transforms.Resize((224, 224)), # 1. Reshape all images to 224x224 (though some models may require different sizes)
                    transforms.ToTensor(), # 2. Turn image values to between 0 & 1 
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], # 3. A mean of [0.485, 0.456, 0.406] (across each colour channel)
                                         std=[0.229, 0.224, 0.225]) # 4. A standard deviation of [0.229, 0.224, 0.225] (across each colour channel),
])
    return transform


def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of training loss and training accuracy metrics.
        In the form (train_loss, train_accuracy). For example:

        (0.1112, 0.8743)
    """
  # Put model in train mode
    model.train()

  # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
    pred_labels = []
    target_labels = []

  # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

        pred_labels = pred_labels + y_pred_class.tolist()
        target_labels = target_labels + y.tolist()
    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    train_f1 = multiclass_f1_score(torch.tensor(pred_labels), torch.tensor(target_labels), num_classes=4)

    return train_loss, train_acc, train_f1



def val_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device):
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
        model: A PyTorch model to be tested.
        dataloader: A DataLoader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to calculate loss on the test data.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of testing loss and testing accuracy metrics.
        In the form (test_loss, test_accuracy). For example:

        (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval() 

    # Setup test loss and test accuracy values
    val_loss, val_acc = 0, 0
    pred_labels = []
    target_labels = []

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            val_pred_logits = model(X)
    # 2. Calculate and accumulate loss
            loss = loss_fn(val_pred_logits, y)
            val_loss += loss.item()

            # Calculate and accumulate accuracy
            val_pred_labels = val_pred_logits.argmax(dim=1)
            val_acc += ((val_pred_labels == y).sum().item()/len(val_pred_labels))
            
            pred_labels = pred_labels + val_pred_labels.tolist()
            target_labels = target_labels + y.tolist()

    # Adjust metrics to get average loss and accuracy per batch 
    val_loss = val_loss / len(dataloader)
    val_acc = val_acc / len(dataloader)

    val_f1 = multiclass_f1_score(torch.tensor(pred_labels), torch.tensor(target_labels), num_classes=4)
    return val_loss, val_acc, val_f1


def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          val_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
    
    results = {"train_loss": [],
      "train_acc": [],
      "train_f1": [],
      "val_loss": [],
      "val_acc": [],
      "val_f1": []
    }

    for epoch in tqdm(range(epochs)):
      train_loss, train_acc, train_f1 = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
      val_loss, val_acc, val_f1 = val_step(model=model,
          dataloader=val_dataloader,
          loss_fn=loss_fn,
          device=device)

      # Print out what's happening
      print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"train_f1: {train_f1:.4f} | "
          f"val_loss: {val_loss:.4f} | "
          f"val_acc: {val_acc:.4f} | "
          f"val_f1: {val_f1:.4f} | "
      )

      # Update results dictionary
      results["train_loss"].append(train_loss)
      results["train_acc"].append(train_acc)
      results["train_f1"].append(train_f1)
      results["val_loss"].append(val_loss)
      results["val_acc"].append(val_acc)
      results["val_f1"].append(val_f1)

  # Return the filled results at the end of the epochs
    return results

    
def test_run(model, test_data, device, batch_size, classes):

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_classcorrect = [0 for i in range(4)] 
        n_classsamples = [0 for i in range(4)]

        for images, labels in test_data:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)

            _, predicted = torch.max(output, 1)
            n_samples += labels.size(0)
            n_correct += (predicted==labels).sum().item()


            for i in range(batch_size):
                label = labels[i]
                pred = predicted[i]
                if label == pred:
                    n_classcorrect[label] += 1
                n_classsamples[label] += 1
            
        acc = (n_correct/n_samples)*100
        print(f"samples accuracy: {acc:.3f}%")

        for i in range(4):
            acc = n_classcorrect[i]/n_classsamples[i] *100
            print(f"Acc for Class {classes[i]} = {acc:.3f}%")
