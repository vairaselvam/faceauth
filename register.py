import face_recognition
import pickle
import os
from PIL import Image
import numpy
from urllib import request
import io

def get_face_embeddings_from_image(image, convert_to_rgb=False):
    print(type(image))
    if convert_to_rgb:
        image = image[:, :, ::-1]

    # run the face detection model to find face locations
    face_locations = face_recognition.face_locations(image)
#     print(face_locations)

    # run the embedding model to get face embeddings for the supplied locations
    face_encodings = face_recognition.face_encodings(image, face_locations)

    return face_locations, face_encodings
#     return face_locations

def register_local_image(locimgpath, identity):
	fn_status = 'fail'
	database_filename = 'database.file'
	if os.path.exists(database_filename):
		with open(database_filename,'rb') as rfp: 
			database = pickle.load(rfp)
	# image1_rgb = face_recognition.load_image_file('./intranet_images/rahul.jpg')
	image = face_recognition.load_image_file(locimgpath)
	# response = request.urlopen(imgurl)
	# image_data = response.read()
	# image_rgb = Image.open(io.BytesIO(image_data))
	# image_numpy = numpy.asarray(image_rgb)
	# image_numpy.flags['WRITEABLE'] = True
	# image_numpy = image_numpy[:,:,0:3]
	# print(image_numpy.shape)
	locations, encodings = get_face_embeddings_from_image(image)
	if len(encodings) == 0:
		print(f'Face encodings not found for user {identity}.')
	else:
		print(f'Encoding face for user: {identity}')
		database[identity] = encodings[0]
	# Now we "sync" our database
	with open(database_filename,'wb') as wfp:
		pickle.dump(database, wfp)
	fn_status = 'success'
	return fn_status

#register_local_image("./trainimages/vaira.jpg", "Vaira200")