# -*- coding: utf-8 -*-
import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader

from Network import Model
from Dataset import CascadeDataset
import os

# Set up device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

def get_max_len(dataset):
    max_len = 0
    for data in tqdm(dataset, desc="Calculating max length"):
        if len(data.x) > max_len:
            max_len = len(data.x)
    return max_len

def msle(y_true, y_pred):
    N = len(y_true)
    sum_error = 0
    for i in range(len(y_true)):
        error = torch.pow((torch.log2(y_pred[i] + 1) - torch.log2(y_true[i] + 1)), 2)
        sum_error += error
    
    res_msle = (1/N) * sum_error
    
    return res_msle

def test_model(model, test_batches):
    """
    Tests the model based on a specific accuracy criterion and calculates the average test loss.
    A prediction is considered correct only if both the predicted value and the true value are greater than 11.
    The accuracy is calculated as the ratio of correct predictions to the total number of predictions that were greater than 11.
    """
    model.eval()
    correct_predictions_over_11 = 0
    total_predictions_over_11 = 0
    both_under_11_count = 0
    test_loss_list = []

    with torch.no_grad():
        for batch in tqdm(test_batches, desc="Testing"):
            batch.to(device)
            pred = model(batch)
            y = batch.y.to(device)

            # Calculate loss for the batch
            loss = msle(y, pred)
            test_loss_list.append(loss.item())

            # Calculate accuracy
            for i in range(len(pred)):
                prediction_value = pred[i].item()
                true_value = y[i].item()

                if prediction_value > 11:
                    total_predictions_over_11 += 1
                    if true_value > 11:
                        correct_predictions_over_11 += 1
                
                if prediction_value < 11 and true_value < 11:
                    both_under_11_count += 1

    if total_predictions_over_11 == 0:
        print("No predictions were over 11. Cannot calculate accuracy.")
        accuracy = 0.0
    else:
        accuracy = (correct_predictions_over_11 / total_predictions_over_11) * 100
    
    avg_loss = sum(test_loss_list) / len(test_loss_list) if test_loss_list else 0

    return accuracy, correct_predictions_over_11, total_predictions_over_11, avg_loss, both_under_11_count

if __name__ == "__main__":
    # --- Configuration ---
    DATASET_ROOT = "data/weibo/weibo_12"
    MODEL_PATH = 'best_model.pt'
    BATCH_SIZE = 64

    # --- Load Dataset ---
    if not os.path.exists(DATASET_ROOT):
        print(f"Error: Dataset directory not found at '{DATASET_ROOT}'")
        exit()
        
    dataset = CascadeDataset(root=DATASET_ROOT)
    
    print("Finding max length for the model...")
    max_len = get_max_len(dataset)
    
    # --- Prepare Test Data ---
    # Use the same split ratio as in main.py to ensure consistency
    N = len(dataset)
    test_start_index = int(N * 0.8)
    test_data = dataset[test_start_index:N]
    
    test_batches = DataLoader(test_data, batch_size=BATCH_SIZE)
    
    # --- Load Model ---
    print(f"Loading the best model from '{MODEL_PATH}'...")
    model = Model(in_channels=4, num_nodes=100, max_len=max_len).to(device)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'. Please run the training script first to generate it.")
        exit()

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Model loaded successfully.")

    # --- Run Test ---
    accuracy, correct_count, total_count, avg_loss, both_under_9_count = test_model(model, test_batches)
    
    # --- Print Results ---
    print("\n--- Test Results ---")
    print(f"Average test loss: {avg_loss:>4f}")
    print(f"Total predictions with value > 11: {total_count}")
    print(f"Correct predictions (prediction > 11 AND ground truth > 11): {correct_count}")
    print(f"Accuracy for predictions > 11: {accuracy:.2f}%")
    print(f"Count where both prediction and ground truth are < 11: {both_under_9_count}")
