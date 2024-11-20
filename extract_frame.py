#Author: Syrine HADDAD
import os
import cv2

def extract_frames(video_path, frame_store_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for subdir, dirs, files in os.walk(video_path):
        for file in files:
            file_path = os.path.join(subdir, file)
            vidcap = cv2.VideoCapture(file_path)
            
            frame_rate = vidcap.get(cv2.CAP_PROP_FPS)  # Get the frame rate of the video
            target_frame_rate = 0.1  # Target frame rate for extraction
            target_duration = 10.0  # Target duration for extraction in seconds

            success, image = vidcap.read()
            count = 0
            frame_num = 1
            current_time = 0.0

            while success and current_time < target_duration:
                current_time = count / frame_rate  # Calculate the current time in seconds
                if current_time >= target_frame_rate * frame_num:
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                    if len(faces) > 0:
                        x, y, w, h = faces[0]  # Assuming the first detected face is the main face
                        face_image = image[y:y + h, x:x + w]
                        resized_face = cv2.resize(face_image, (256, 256))  # Resize the face image

                        # Get participant ID and emotion ID
                        participant_id = file[:4].lower()
                        emotion_id = file[5:7]

                        frame_filename = os.path.join(frame_store_path, f"{participant_id}_{emotion_id}_frame{frame_num}.jpg")
                        cv2.imwrite(frame_filename, resized_face)
                        frame_num += 1
                
                success, image = vidcap.read()
                count += 1

# Provide the paths to your input videos and the directory where frames should be stored
video_path = "./Cropped_Videos"
frame_store_path = "./frames"

extract_frames(video_path, frame_store_path)
