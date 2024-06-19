import cv2
import mediapipe as mp
import csv
from Calculate_Angle import calculate_angle




mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
video_path = (r"C:\Users\gaith\Desktop\Mussuab\Mussab.webm")

# Define the useful keypoints for the push-up exercise
useful_keypoints = [
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_ANKLE
]

cap = cv2.VideoCapture(video_path)
keypoints_data = []
angles_data = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        frame_keypoints = []
        for keypoint in useful_keypoints:
            landmark = results.pose_landmarks.landmark[keypoint]
            frame_keypoints.append((landmark.x, landmark.y, landmark.z))

        # Calculate mid-hip as the average of left hip and right hip
        mid_hip = (
            (frame_keypoints[6][0] + frame_keypoints[7][0]) / 2,
            (frame_keypoints[6][1] + frame_keypoints[7][1]) / 2,
            (frame_keypoints[6][2] + frame_keypoints[7][2]) / 2
        )
        #
        # Calculate angles
        right_elbow_right_shoulder_right_hip = calculate_angle(frame_keypoints[3], frame_keypoints[1], frame_keypoints[7])
        left_elbow_left_shoulder_left_hip = calculate_angle(frame_keypoints[2], frame_keypoints[0], frame_keypoints[6])
        right_knee_mid_hip_left_knee = calculate_angle(frame_keypoints[9], mid_hip, frame_keypoints[8])
        right_hip_right_knee_right_ankle = calculate_angle(frame_keypoints[7], frame_keypoints[9], frame_keypoints[11])
        left_hip_left_knee_left_ankle = calculate_angle(frame_keypoints[6], frame_keypoints[8], frame_keypoints[10])
        right_wrist_right_elbow_right_shoulder = calculate_angle(frame_keypoints[5], frame_keypoints[3], frame_keypoints[1])
        left_wrist_left_elbow_left_shoulder = calculate_angle(frame_keypoints[4], frame_keypoints[2], frame_keypoints[0])

        angles_data.append([
            right_elbow_right_shoulder_right_hip,
            left_elbow_left_shoulder_left_hip,
            right_knee_mid_hip_left_knee,
            right_hip_right_knee_right_ankle,
            left_hip_left_knee_left_ankle,
            right_wrist_right_elbow_right_shoulder,
            left_wrist_left_elbow_left_shoulder
        ])

cap.release()

# Save the angles data to a CSV file
csv_file_path = (r"C:\Users\gaith\Desktop\Mussuab\Mussab_keypoint_py.csv")
headers = [
    'right_elbow_right_shoulder_right_hip',
    'left_elbow_left_shoulder_left_hip',
    'right_knee_mid_hip_left_knee',
    'right_hip_right_knee_right_ankle',
    'left_hip_left_knee_left_ankle',
    'right_wrist_right_elbow_right_shoulder',
    'left_wrist_left_elbow_left_shoulder',
    'ml_class'
]
with open(csv_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(headers)  # Write the headers
    writer.writerows(angles_data)  # Write the data

print(f"Angles calculated and saved to {csv_file_path}")
