import torch
import os
import torch.nn as nn
import torchvision.transforms as T
from transformers import ViTForImageClassification, ViTConfig, ViTImageProcessor, DeiTForImageClassification
from transformers import DeiTImageProcessor, AutoFeatureExtractor, AutoImageProcessor, AutoModelForImageClassification
from transformers import AutoModel, AutoConfig, ViTMAEModel
from transformers import Dinov2WithRegistersModel,Dinov2WithRegistersConfig, Dinov2WithRegistersForImageClassification #Dinov2WithRegistersModel requires transformers==4.48
from torchvision.transforms import ToPILImage
from torch.nn.functional import cosine_similarity
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

class SwinV2Classifier(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.6, pretrained_model_name="microsoft/swinv2-tiny-patch4-window8-256"):
        super(SwinV2Classifier, self).__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the processor for image preprocessing
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
        
        # Load the pretrained SwinV2 Transformer model
        self.swinv2 = AutoModelForImageClassification.from_pretrained(
            pretrained_model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            output_attentions=True  # Enable attention outputs for interpretability
        )
        
        # Replace the classifier with a custom head
        self.swinv2.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.swinv2.config.hidden_size, num_classes)
        )
    
    def forward(self, x):
        # Convert tensors to PIL images for compatibility with the processor
        to_pil = ToPILImage()
        images = [to_pil(img) for img in x]

        # Preprocess images with the processor
        inputs = self.processor(images=images, return_tensors="pt").pixel_values.to(self.device)

        # Forward pass through the model
        outputs = self.swinv2(pixel_values=inputs)
        return outputs.logits, outputs.attentions  # Return logits and attention weights

class DINOv2WithRegistersClassifier(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.6, pretrained_model_name="facebook/dinov2-with-registers-small"):
        super(DINOv2WithRegistersClassifier, self).__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the processor for image preprocessing
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
        self.processor.do_rescale = False  # Avoid double rescaling
        
        # Load the model configuration and enable attention outputs
        config = Dinov2WithRegistersConfig.from_pretrained(pretrained_model_name)
        config.output_attentions = True
        
        # Load the pretrained DINOv2 with registers model
        self.dinov2 = Dinov2WithRegistersModel.from_pretrained(
            pretrained_model_name, config=config, attn_implementation="eager"
        )
                
        # Add a classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.dinov2.config.hidden_size, num_classes)
        )
    
    def forward(self, x, output_attentions=True):
        # Normalize inputs to [0, 1] if necessary
        if x.min() < 0 or x.max() > 1:
            x = torch.clamp((x - x.min()) / (x.max() - x.min()), 0, 1)
        
        # Preprocess images directly
        inputs = self.processor(images=x, return_tensors="pt").pixel_values.to(self.device)

        # Forward pass through the base model
        outputs = self.dinov2(
            pixel_values=inputs,
            output_attentions=output_attentions
        )

        # Use mean pooling for token embeddings
        last_hidden_state = outputs.last_hidden_state
        pooled_embeddings = last_hidden_state.mean(dim=1)

        # Pass through the classification head
        logits = self.classifier(pooled_embeddings)
        attentions = outputs.attentions if output_attentions else None
        return logits, attentions

class MAEClassifier(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.6, pretrained_model_name="facebook/vit-mae-base"):
        super(MAEClassifier, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load image processor for preprocessing spectrograms
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)

        # Load pre-trained MAE encoder
        self.mae = ViTMAEModel.from_pretrained(pretrained_model_name)

        # Add classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.mae.config.hidden_size, num_classes)  # Fully connected layer for classification
        )

    def forward(self, x, output_attentions=True):
        # Convert tensors to PIL images for compatibility with processor
        to_pil = ToPILImage()
        images = [to_pil(img) for img in x]

        # Preprocess images using Hugging Face processor
        inputs = self.processor(images=images, return_tensors="pt").pixel_values.to(self.device)

        # Pass through MAE model
        outputs = self.mae(pixel_values=inputs)

        # Extract CLS token representation for classification
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Pass through the classifier head
        logits = self.classifier(cls_embedding)
        attentions = outputs.attentions if output_attentions else None

        return logits, attentions

# To train and save the model after training
def train_and_save(model, train_loader, 
                   eval_loader, num_epochs, lr=0.001, save_path="model.pth", 
                   patience=3, weight_decay=0.01,
                   pretrain_epochs=50, 
                   teacher_model=None,  
                   mode="supervised",  # "pretrain", "supervised", or "both",
                   datasets_name=None
                   ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

   # Convert list to string for display
    datasets_str = ", ".join(datasets_name) if datasets_name else "Unknown Dataset"

    print(f"Training Model: {model.__class__.__name__}")
    print(f"Using Datasets: {datasets_str}")  # Print dataset names

    class_weights = torch.tensor([1.0, 1.0, 0.25, 1.0]).to(device) 
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, verbose=True)
    
    if mode in ["pretrain", "both"]:
        if teacher_model is None:
            raise ValueError("Teacher model must be provided for unsupervised pretraining.")
        
        print("\nStarting Unsupervised Pretraining...")
        pretrain_optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        for epoch in range(pretrain_epochs):
            model.train()
            total_loss = 0.0
            for images, _ in train_loader:
                images = images.to(device)

                # Generate augmentations
                weak_augmented = weak_augmentation(images)
                strong_augmented = strong_augmentation(images)

                # Teacher output (extract embeddings or logits)
                teacher_logits, _ = teacher_model(strong_augmented)  # Assuming (logits, attentions) is returned
                teacher_output = teacher_logits.detach()  # Detach to prevent gradient flow

                # Student output
                student_logits, _ = model(weak_augmented)  # Assuming (logits, attentions) is returned
                student_output = student_logits

                # Compute self-supervised loss
                loss = -cosine_similarity(teacher_output, student_output, dim=-1).mean()

                pretrain_optimizer.zero_grad()
                loss.backward()
                pretrain_optimizer.step()

                total_loss += loss.item()

            print(f"Unsupervised Pretrain Epoch [{epoch + 1}/{pretrain_epochs}], Loss: {total_loss / len(train_loader):.4f}")

        print("\nUnsupervised Pretraining Completed. Saving pretrained weights...")
        torch.save(model.state_dict(), save_path.replace(".pth", "_pretrained.pth"))
    
        if mode == "pretrain":
            print("Exiting after pretraining.")
            return  # Skip supervised training if only pretraining is required

    # Supervised Training
    if mode in ["supervised", "both"]:
        print("\nStarting Supervised Training (Vision Transformer)...")
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
        
        print("Supervised Training Complete.")
        return results

# To reload the trained model later
def load_trained_model(model_class, save_path, num_classes=4, pretrained=False):
        # Avoid appending "_pretrained" multiple times
    if pretrained and not save_path.endswith("_pretrained.pth"):
        save_path = save_path.replace(".pth", "_pretrained.pth")

    # Check if the checkpoint exists
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Checkpoint file not found: {save_path}")

    # Instantiate and load the model
    model = model_class(num_classes=num_classes)
    state_dict = torch.load(save_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded {'pretrained' if pretrained else 'fine-tuned'} model from: {save_path}")
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

def weak_augmentation(images):
    return T.Compose([
        T.RandomAffine(degrees=5),
        T.RandomHorizontalFlip(p=0.5),
    ])(images)

def strong_augmentation(images):
    return T.Compose([
        T.RandomAffine(degrees=30),
        T.ColorJitter(brightness=0.3, contrast=0.3),
        T.RandomErasing(p=0.4, scale=(0.02, 0.1)),
    ])(images)