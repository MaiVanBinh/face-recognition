
import os
import sys
sys.path.append('../insightface/deploy')
sys.path.append('../insightface/src/common')

import cv2

from mtcnn.mtcnn import MTCNN
import face_preprocess
import numpy as np
import argparse


ap = argparse.ArgumentParser()

ap.add_argument("--output", default="../datasets/unlabeled_faces",
                help="Path to faces output")
ap.add_argument("--image", default="../datasets/test/001.jpg",
    help='Path to output image')

args = vars(ap.parse_args())

# Detector = mtcnn_detector
detector = MTCNN()

# Read Imgae
frame = cv2.imread(args["image"])

faces = 0

# if frame%10 == 0:
    # Get all faces on current frame
bboxes = detector.detect_faces(frame)

if len(bboxes) != 0:
    # Get only the biggest face
    for bboxe in bboxes:
        bbox = bboxe["box"]
        bbox = np.array([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
        landmarks = bboxe["keypoints"]

        # convert to face_preprocess.preprocess input
        landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0], landmarks["mouth_left"][0], landmarks["mouth_right"][0],
                                landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1], landmarks["mouth_left"][1], landmarks["mouth_right"][1]])
        landmarks = landmarks.reshape((2,5)).T
        nimg = face_preprocess.preprocess(frame, bbox, landmarks, image_size='112,112')
        if not(os.path.exists(args["output"])):
            os.makedirs(args["output"])
        # cv2.imwrite(os.path.join(args["output"], "{}.jpg".format(faces+1)), nimg)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        print("[INFO] {} faces detected".format(faces+1))
        faces += 1
cv2.imshow("image", frame)
cv2.waitKey(0)
