from ultralytics import YOLO

# Load trained model
model = YOLO('runs/classify/train/weights/best.pt')

# Predict on the test folder
results = model.predict(source='C:/Users/iC/Desktop/project/dataset/val', save=True)  # save=True saves predicted results

# Print summary
print('Predictions done!')
