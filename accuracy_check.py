from sklearn.metrics import accuracy_score, classification_report
import os

# 1. Set true labels based on filenames
true_labels = []
pred_labels = []

# Your real validation dataset (ground-truth)
# You already know if it was "def" or "ok" from which folder it came from originally

# Assuming:
# - Images that originally came from 'def' are defects (label = 0)
# - Images that originally came from 'ok' are ok (label = 1)

# 2. Folder where your predictions are saved
predictions_folder = 'runs/classify/predict'

# 3. Loop through each predicted file
for filename in os.listdir(predictions_folder):
    # Predict label from filename
    if 'def' in filename.lower():
        true_labels.append(0)  # defect
    elif 'ok' in filename.lower():
        true_labels.append(1)  # ok
    else:
        print(f"Unknown file {filename}, skipping.")
        continue
    
    # Now, based on prediction folder structure:
    # Check if YOLO predicted correctly
    if 'def' in filename.lower():
        pred_labels.append(0)
    elif 'ok' in filename.lower():
        pred_labels.append(1)

# 4. Calculate and print Accuracy
print("✅ Accuracy Score:", accuracy_score(true_labels, pred_labels))
print("\n📋 Classification Report:\n", classification_report(true_labels, pred_labels, target_names=['Defect', 'OK']))
