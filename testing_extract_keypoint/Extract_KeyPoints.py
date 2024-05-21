import cv2
import mediapipe as mp
import csv
i = 0
while(i<=99):
    number_of_json_file = i


    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    video_path = f"C:\\Users\\gaith\\Desktop\\Final Project23-24\\Datasets\\InfinityAI_InfiniteRep_pushup_v1.0\\data\\00000{number_of_json_file}.mp4"

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

    keypoint_names = [
        "LEFT_SHOULDER",
        "RIGHT_SHOULDER",
        "LEFT_ELBOW",
        "RIGHT_ELBOW",
        "LEFT_WRIST",
        "RIGHT_WRIST",
        "LEFT_HIP",
        "RIGHT_HIP",
        "LEFT_KNEE",
        "RIGHT_KNEE",
        "LEFT_ANKLE",
        "RIGHT_ANKLE"
    ]

    cap = cv2.VideoCapture(video_path)
    keypoints_data = []

    # Create header for the CSV file
    header = []
    for name in keypoint_names:
        header.extend([f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_visibility"])

    keypoints_data.append(header)

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
                frame_keypoints.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            keypoints_data.append(frame_keypoints)

    cap.release()

    # Save the keypoints data to a CSV file
    csv_file_path =f"{number_of_json_file}  JsonFile.csv"
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(keypoints_data)

    print(f"Keypoints extracted and saved to {csv_file_path}")
    i += 1







