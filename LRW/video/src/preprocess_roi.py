""" Extracts ROI from video frames and saves them as numpy arrays. """

import os
import glob
from tqdm import tqdm
import numpy as np
import cv2
import mediapipe as mp
from multiprocessing import Pool, cpu_count


def read_landmark(path):
    list_frames = []
    list_landmarks = []
    
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        frame_idx = 0
        cap = cv2.VideoCapture(path)
        while cap.isOpened():
            ret, frame = cap.read()  # BGR
            if ret:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = face_mesh.process(image)
            else:
                break

            list_landmarks = []
            # accumulate data
            if results.multi_face_landmarks is not None:
                for landmark in zip(results.multi_face_landmarks[0].landmark):
                    list_landmarks.append([landmark[0].x, landmark[0].y, landmark[0].z])
                prev_landmarks = list_landmarks  # update
            elif results.multi_face_landmarks is None:
                if frame_idx == 0:
                    list_landmarks = np.zeros((478, 3)).tolist()
                    prev_landmarks = list_landmarks  # update
                elif frame_idx > 0:
                    list_landmarks = prev_landmarks  # inherit
            list_frames.append(list_landmarks)
            frame_idx += 1
    path = path.replace(".mp4", ".npy")
    np.save(path, np.array(list_frames))
    print("Finished processing {}".format(path))


if __name__ == "__main__":
    root_dir = "./LRW/dset_mp4/"
    
    with open("./LRW/labels.txt") as myfile:
        labels = myfile.read().splitlines()

    all_files = []
    for label in tqdm(labels):
        word_label_files = glob.glob(os.path.join(root_dir, label, "*", "*.mp4"))
        word_label_files.sort()
        all_files.extend(word_label_files)

    print(f"Total number of files: {len(all_files)} with {len(labels)} labels")

    num_workers = cpu_count()
    pool = Pool(processes=num_workers)
    
    # sort folders
    pool.map(read_landmark, all_files)

