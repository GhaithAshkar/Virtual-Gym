import json
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
number_of_json_file = "01"
import cv2

path = fr"C:\Users\gaith\Desktop\Final Project23-24\Datasets\InfinityAI_InfiniteRep_pushup_v1.0\data\0000{number_of_json_file}.json"
with open(path) as f:
    data = json.load(f)

# Step 2: Create a dictionary to map image_id to rep_count
image_id_to_rep_count = {image['id']: image['rep_count'] for image in data['images']}

# Step 3: Extract keypoints and rep_count from the JSON data
keypoints = []
rep_counts = []

for annotation in data['annotations']:
    if 'keypoints' in annotation:
        keypoints.append(annotation['keypoints'])
        image_id = annotation['image_id']
        rep_count = image_id_to_rep_count[image_id]
        rep_counts.append(rep_count)
     #   print(keypoints)
     #   print(rep_counts)
# Step 4: Save keypoints and rep_count to a CSV file
with open(f'0{number_of_json_file}Json.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['keypoints', 'rep_count'])
    for kp, rc in zip(keypoints, rep_counts):
        writer.writerow([kp, rc])
# Step 5: Load the CSV file and preprocess the data
df = pd.read_csv(f'0{number_of_json_file}Json.csv')









# import cv2
# import mediapipe as mp
# import numpy as np
# import pandas as pd
#
# # TODO 1 :   Initialize MediaPipe Pose Model
#
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
#
# # TODO 2 : Extract Body Keypoints
#
# def extract_keypoints(image_path):
#     image = cv2.imread(image_path)
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = pose.process(image_rgb)
#
#     keypoints = []
#     if results.pose_landmarks:
#         for landmark in results.pose_landmarks.landmark:
#             keypoints.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
#
#     return keypoints


