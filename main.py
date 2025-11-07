import cv2
from fer import fer
import matplotlib.pyplot as plt

# Load an example image (replace this with any image path)
img_path = "grup.jpg"  # Put a local photo here
img = cv2.imread(img_path)

# Initialize detector
detector = fer.FER(mtcnn=True)
# Detect emotions
results = detector.detect_emotions(img)

# Print results
for r in results:
    print(f"Detected emotions: {r['emotions']}")
    print(f"Dominant emotion: {detector.top_emotion(img)}")

# Show image with detections
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
