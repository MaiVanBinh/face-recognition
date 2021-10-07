
import os
import sys
import constants

sys.path.append('../insightface/deploy')
sys.path.append('../insightface/src/common')

import cv2
from mtcnn.mtcnn import MTCNN
import face_preprocess
import numpy as np

def get_faces_from_image(imagePath, nameFolder):
    '''
        Get faces in new image and store list faces in personal folder
        imagePath: path of new image
        nameFolder: name folder to store list faces
    '''
    outPath = constants.FACE_REGISTER_FOR_TRAIN_PATH + '/' + nameFolder 

    # Detector = mtcnn_detector
    detector = MTCNN()

    # Read Imgae
    frame = cv2.imread(imagePath)

    faces = 0

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
            if not(os.path.exists(outPath)):
                os.makedirs(outPath)
            cv2.imwrite(os.path.join(outPath, "{}.jpg".format(faces+1)), nimg)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            # print(bbox[1], bbox[3], bbox[0],bbox[2])
            faces += 1

