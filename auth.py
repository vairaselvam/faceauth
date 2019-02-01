import face_recognition
import cv2
import numpy as np
import glob
import os
import logging
import pickle

IMAGES_PATH = './intranet_images'  # put your reference images in here
#IMAGES_PATH = './images'    # put your reference images in here
CAMERA_DEVICE_ID = 0
MAX_DISTANCE = 0.6  # increase to make recognition less strict, decrease to make more strict

def get_face_embeddings_from_image(image, convert_to_rgb=False):
    """
    Take a raw image and run both the face detection and face embedding model on it
    """
    # Convert from BGR to RGB if needed
    if convert_to_rgb:
        image = image[:, :, ::-1]

    # run the face detection model to find face locations
    face_locations = face_recognition.face_locations(image)
#     print(face_locations)

    # run the embedding model to get face embeddings for the supplied locations
    face_encodings = face_recognition.face_encodings(image, face_locations)

    return face_locations, face_encodings

def setup_database():
    """
    Load reference images and create a database of their face encodings
    """
    database = {}
    count = 0
    print(os.path.join(IMAGES_PATH, '*.jpg'))
    for filename in glob.glob(os.path.join(IMAGES_PATH, '*.jpg')):
        # load image
        print(filename)
        image_rgb = face_recognition.load_image_file(filename)

        # use the name in the filename as the identity key
        identity = os.path.splitext(os.path.basename(filename))[0]

        # get the face encoding and link it to the identity
        locations, encodings = get_face_embeddings_from_image(image_rgb)
        print(locations)
#         print(encodings)
        if len(encodings) == 0:
            print(f'Face encodings not found for user {identity}.')
        else:
            print(f'Encoding face for user #{count}: {identity}')
            database[identity] = encodings[0]
            count = count + 1
    db_file_name = 'database.file'
    with open(db_file_name, 'wb') as fp:
        pickle.dump(database, fp)
    return database

# Required only once
#database = setup_database()

def load_database_from_file():
    db_file_name = 'database.file'
    with open(db_file_name, 'rb') as fp:
        database = pickle.load(fp)

    return database

def run_face_recognition_image_local(face_img):

    database = load_database_from_file()
    """
    Start the face recognition from an Image
    """
    # the face_recognitino library uses keys and values of your database separately
    known_face_encodings = list(database.values())
    known_face_names = list(database.keys())

    pil_im = cv2.imread(face_img)
    col_img = cv2.cvtColor(pil_im, cv2.COLOR_BGR2RGB)
#     plt.figure(figsize=(9,6))
#     plt.imshow(col_img)
#     plt.show()  
    #convert the test image to gray image as opencv face detector expects gray images 

    frame = col_img.copy()

    # run detection and embedding models
    face_locations, face_encodings = get_face_embeddings_from_image(frame, convert_to_rgb=True)

    # Loop through each face in this frame of video and see if there's a match
    for location, face_encoding in zip(face_locations, face_encodings):

        # get the distances from this encoding to those of all reference images
        distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        # select the closest match (smallest distance) if it's below the threshold value
        if np.any(distances <= MAX_DISTANCE):
            best_match_idx = np.argmin(distances)
            name = known_face_names[best_match_idx]
        else:
            name = None

    return name
        # put recognition info on the image
#         paint_detected_face_on_image(frame, location, name)
#     plt.figure(figsize=(15,12))
#     plt.imshow(frame)
#     plt.show()


#print(list(database.keys()))
#name = run_face_recognition_image_local('vaira.jpg')
#print(name)
