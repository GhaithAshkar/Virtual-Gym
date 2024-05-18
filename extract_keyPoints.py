import cv2
import mediapipe as mp
import csv

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

video_path = "C:\\Users\\gaith\\Desktop\\Final Project23-24\\Datasets\\InfinityAI_InfiniteRep_pushup_v1.0\\data\\000000.mp4"
cap = cv2.VideoCapture(video_path)

keypoints_data = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform pose estimation
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        # Extract keypoints
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z])

        # Append the keypoints to the data list
        keypoints_data.append(keypoints)

cap.release()
pose.close()

# Save the keypoints to a CSV file
csv_file_path = "keypoints.csv"
with open(csv_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(keypoints_data)

print(f"Keypoints extracted and saved to {csv_file_path}")