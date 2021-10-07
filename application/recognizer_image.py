import constants
import face_model
import numpy as np
import face_preprocess
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
import cv2
import pickle
import sys
sys.path.append('../insightface/deploy')
sys.path.append('../insightface/src/common')


def findCosineDistance(vector1, vector2):
    """
    Calculate cosine distance between two vector
    """
    vec1 = vector1.flatten()
    vec2 = vector2.flatten()

    a = np.dot(vec1.T, vec2)
    b = np.dot(vec1.T, vec1)
    c = np.dot(vec2.T, vec2)
    return 1 - (a/(np.sqrt(b)*np.sqrt(c)))


def CosineSimilarity(test_vec, source_vecs):
    """
    Verify the similarity of one vector to group vectors of one class
    """
    cos_dist = 0
    for source_vec in source_vecs:
        cos_dist += findCosineDistance(test_vec, source_vec)
    return cos_dist/len(source_vecs)


def recognizer_image(pathImageIn, pathImageOut):
    '''
        return name and location of face
    '''
    detectorPerson = []

    args = {
        "mymodel": constants.FACE_MODEL_PATH,  # Path to recognizer model
        "le": constants.FACE_DATA_LABEL_PATH,  # Path to label encoder
        "embeddings": constants.FACE_DATA_VECTOR_PATH,  # Path to embeddings
        "image_out": pathImageOut,
        "image_in": pathImageIn,
        "image_size": constants.FACE_INPUT_IMAGE_SIZE,
        "model": constants.FACE_MODEL_CONFIGURATION,  # path to load model
        "ga_model": '',  # path to load model
        "gpu": 0,  # gpu id
        "det": 0,  # mtcnn option, 1 means using R+O, 0 means detect from begining
        "flip": 0,  # whether do lr flip aug
        "threshold": 1.24  # ver dist threshold
    }

    # Load embeddings and labels
    data = pickle.loads(open(args["embeddings"], "rb").read())
    le = pickle.loads(open(args["le"], "rb").read())

    embeddings = np.array(data['embeddings'])
    labels = le.fit_transform(data['names'])

    # Initialize detector
    detector = MTCNN()

    # Initialize faces embedding model
    embedding_model = face_model.FaceModel(args)

    # Load the classifier model
    model = load_model(args["mymodel"])

    # Define distance function

    # Setup some useful arguments
    cosine_threshold = 0.8
    proba_threshold = 0.85
    comparing_num = 5

    img = cv2.imread(args["image_in"])

    bboxes = detector.detect_faces(img)

    if len(bboxes) != 0:
        for bboxe in bboxes:
            bbox = bboxe['box']
            bbox = np.array(
                [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
            landmarks = bboxe['keypoints']
            landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0], landmarks["mouth_left"][0], landmarks["mouth_right"][0],
                                  landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1], landmarks["mouth_left"][1], landmarks["mouth_right"][1]])
            landmarks = landmarks.reshape((2, 5)).T
            nimg = face_preprocess.preprocess(
                img, bbox, landmarks, image_size='112,112')
            nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
            nimg = np.transpose(nimg, (2, 0, 1))
            embedding = embedding_model.get_feature(nimg).reshape(1, -1)

            text = "Unknown"

            # Predict class
            preds = model.predict(embedding)
            preds = preds.flatten()
            # Get the highest accuracy embedded vector
            j = np.argmax(preds)
            proba = preds[j]
            # Compare this vector to source class vectors to verify it is actual belong to this class
            match_class_idx = (labels == j)
            match_class_idx = np.where(match_class_idx)[0]
            selected_idx = np.random.choice(match_class_idx, comparing_num)
            compare_embeddings = embeddings[selected_idx]
            # Calculate cosine similarity
            cos_similarity = CosineSimilarity(embedding, compare_embeddings)
            if cos_similarity < cosine_threshold and proba > proba_threshold:
                name = le.classes_[j]
                text = "{}".format(name)
                # print("Recognized: {} <{:.2f}>".format(name, proba*100))

            y = bbox[1] - 10 if bbox[1] - 10 > 10 else bbox[1] + 10
            cv2.putText(img, text, (bbox[0], y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(img, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]), (255, 0, 0), 2)
            detectorPerson.append({"name": text,  "startPoint": [
                                  bbox[0], bbox[1]], "endPoint": [bbox[2], bbox[3]]})
    cv2.imwrite(args["image_out"], img)
    return detectorPerson
