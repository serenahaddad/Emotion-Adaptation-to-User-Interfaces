#Author: Syrine HADDAD
import torch
import torchvision.transforms as transforms
import cv2
from PIL import Image
import numpy as np
from only_vid_classifier2 import Net

# Load the pretrained model
model = Net()
model.load_state_dict(torch.load('GUImodel.pth', map_location=torch.device('cpu')))
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to perform inference on a single image
def predict_emotion(image):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # Preprocess the image
    image = Image.fromarray(image)
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)[0]  # Get softmax probabilities
        _, predicted = torch.max(output, 1)
        emotion = ['Happy', 'Angry', 'Calm','Surprise'][predicted.item()]
        return emotion, probabilities

# Open a webcam
cap = cv2.VideoCapture(1)

# Load the face detection cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from webcam")
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Crop the face region for emotion prediction
        face_roi = gray[y:y+h, x:x+w]

        # Perform emotion prediction on the face region
        emotion, probabilities = predict_emotion(face_roi)

        # Display the predicted emotion above the rectangle
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display probabilities in a separate text box
        prob_text = 'Probabilities:\n'
        emotions = ["Happy", "Angry", "Calm", "Surprise"]
        for i, prob in enumerate(probabilities):
            prob_text += f'{emotions[i]}: {prob.item():.2f}\n'

        text_color = (0, 0, 255)  # Red color
        text_position = (10, 50)  # Starting position for text

        # Display each emotion and its prediction on a separate line
        for i, line in enumerate(prob_text.split('\n')):
            cv2.putText(frame, line, (text_position[0], text_position[1] + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    # Display the frame
    cv2.imshow('Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
