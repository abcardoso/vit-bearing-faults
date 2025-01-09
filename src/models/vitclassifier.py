import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig, ViTImageProcessor, DeiTForImageClassification, DeiTImageProcessor, AutoFeatureExtractor
from torchvision.transforms import ToPILImage
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


class ViTClassifier(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.6):
        super(ViTClassifier, self).__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the processor for preprocessing images
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.processor.do_rescale = False

        # Load the ViT model with the correct number of classes, replacing the classifier
        self.vit = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=num_classes,
            output_attentions=True, # Enable attention outputs
            ignore_mismatched_sizes=True,  # To handle mismatched classifier layer size
            attn_implementation="eager"  # Explicitly specify the attention implementation
        )
        
        # Manually adjust the classifier layer if necessary
        #self.vit.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)
        self.vit.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),  # Add Dropout before the final layer
            nn.Linear(self.vit.config.hidden_size, num_classes)
        )


    def forward(self, x):
        # Convert tensors back to PIL images for compatibility with ViTImageProcessor
        to_pil = ToPILImage()
        images = [to_pil(img) for img in x]

        # Preprocess images with the processor for ViT compatibility
        inputs = self.processor(images=images, return_tensors="pt").pixel_values.to(x.device)

        # Forward pass through the model
        output = self.vit(pixel_values=inputs)
        return output.logits, output.attentions  # Return logits and attention weights
 
class DeiTClassifier(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.6):
        super(DeiTClassifier, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # # Correct initialization for DeiT model
        # self.processor = DeiTImageProcessor.from_pretrained("facebook/deit-base-patch16-224")
        # self.processor.do_rescale = False
        # Feature extractor
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/deit-base-patch16-224")

        # Ensure only the correct model is loaded
        self.deit = ViTForImageClassification.from_pretrained(
            "facebook/deit-base-patch16-224",
            num_labels=num_classes,
            output_attentions=True,
            attn_implementation="eager",
            ignore_mismatched_sizes=True
        )

        # Custom classifier definition
        self.deit.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.deit.config.hidden_size, num_classes)
        )

    def forward(self, x):
        to_pil = ToPILImage()
        images = [to_pil(img) for img in x]
        inputs = self.feature_extractor(images=images, return_tensors="pt").pixel_values.to(self.device)

        output = self.deit(pixel_values=inputs)
        return output.logits, output.attentions
    
# To train and save the model after training
def train_and_save(model, train_loader, eval_loader, num_epochs, lr=0.001, save_path="deit_classifier.pth", patience=3, weight_decay=0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"Training Model: {model.__class__.__name__}")

    class_weights = torch.tensor([1.0, 1.0, 0.25, 1.0]).to(device) 
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, verbose=True)
    
    best_loss = float("inf")  
    patience_counter = 0
    
    results = {
        "epoch": [],
        "train_loss": [],
        "eval_loss": [],
        "eval_accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": []
    }
    
    print('Starting Pre Training (Vision Transformer)...')    
    print(f"LR: {lr} | Num Epochs: {num_epochs} | Weight Decay: {weight_decay} | Early Stopping Patience: {patience}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_labels = []
        all_predictions = []
        
        # Training Loop
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            logits, _ = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Collect predictions and labels for metrics calculation
            _, preds = torch.max(logits, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())

        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = accuracy_score(all_labels, all_predictions) * 100

        # Evaluation Phase
        #eval_loss, eval_accuracy = evaluate_model(model, eval_loader, criterion, device)

        # Calculate Metrics
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0) * 100
        recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0) * 100
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0) * 100
        
        results["epoch"].append(epoch + 1)
        results["train_loss"].append(avg_train_loss)
        # results["eval_loss"].append(eval_loss)
        # results["eval_accuracy"].append(eval_accuracy)
        results["precision"].append(precision)
        results["recall"].append(recall)
        results["f1_score"].append(f1)
        
        print(f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Train Accuracy: {train_accuracy:.4f} | "
            #   f"Eval Loss: {eval_loss:.4f} | Eval Accuracy: {eval_accuracy:.2f}% | "
              f"Precision: {precision:.2f}% | Recall: {recall:.2f}% | F1-Score: {f1:.2f}%")

        # Step the scheduler with the evaluation loss
        scheduler.step(avg_train_loss)

        # Early Stopping Check
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            #print(f"Model improved and saved at {save_path}.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered. No improvement in evaluation loss.")
                break

    print("Training completed successfully." if patience_counter < patience else "Training stopped early.")
    return results


# To reload the trained model later
def load_trained_model(model_class, save_path, num_classes=4):
    # Instantiate model with specified num_classes
    model = model_class(num_classes=num_classes) 
    model.load_state_dict(torch.load(save_path, weights_only=True))  
    
    print(f"Load - Training Model: {model.__class__.__name__}")
    
    return model

def evaluate_model(model, eval_loader, criterion, device):
    model.eval()
    eval_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(eval_loader):
            images, labels = images.to(device), labels.to(device)

            logits, _ = model(images)
            loss = criterion(logits, labels)
            eval_loss += loss.item()

            # Collect predictions
            _, preds = torch.max(logits, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())

            # Print batch-level evaluation information
            print(f"[Batch {batch_idx+1}/{len(eval_loader)}] "
                  f"Batch Loss: {loss.item():.4f} | Batch Accuracy: {accuracy_score(labels.cpu(), preds.cpu())*100:.2f}%")

    # Compute average loss and overall accuracy
    avg_loss = eval_loss / len(eval_loader)
    accuracy = accuracy_score(all_labels, all_predictions) * 100

    print(f"[Eval] Average Loss: {avg_loss:.4f} | Overall Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy