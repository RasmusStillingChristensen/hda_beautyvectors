from typing import Tuple
from enum import IntEnum
import cv2
import numpy as np
import os
import dlib
import pickle
from sklearn import svm

import matplotlib.pyplot as plt

B_model = None
FR_model = None
SVM_model = None
landmark_predictor = None

class ReturnCode(IntEnum):
    Success=0
    UnknownError=1
    ConfigError=2
    RefuseInput=3
    ExtractError=4
    ParseError=5
    TemplateCreationError=6
    VerifTemplateError=7
    FaceDetectionError=8
    NumDataError=9
    TemplateFormatError=10
    EnrollDirError=11
    InputLocationError=12
    MemoryError=13
    MatchError=14
    QualityAssessmentError=15
    NotImplemented=16
    VendorError=17

class ImageLabel(IntEnum):
	Unknown=0
	NonScanned=1
	Scanned=2

def initialize(alg_extra_data_folder_path: str) -> ReturnCode:
	return_code = ReturnCode.Success
	global B_model, FR_model, SVM_model, landmark_predictor

	try:
		# Construct paths to the models
		b_model_path = os.path.join(alg_extra_data_folder_path, "VGG16_512.onnx")
		fr_model_path = os.path.join(alg_extra_data_folder_path, "glintr100.onnx")
		svm_model_path = os.path.join(alg_extra_data_folder_path, "SVM_FERET_FRGC_MSYNM.pkl")
		landmark_predictor_path = os.path.join(alg_extra_data_folder_path, "shape_predictor_68_face_landmarks.dat")
        # Load the models
		B_model = cv2.dnn.readNetFromONNX(b_model_path)
		FR_model = cv2.dnn.readNetFromONNX(fr_model_path)
		with open(svm_model_path, 'rb') as file:
			SVM_model = pickle.load(file)
		landmark_predictor = dlib.shape_predictor(landmark_predictor_path)
	except Exception as e:
		return_code = ReturnCode.InputLocationError
	return return_code

def preprocess_image_arcface(image):
	input_mean = 127.5
	input_std = 127.5
	input_size = (112, 112)
	blob = cv2.dnn.blobFromImages([image], 1.0 / input_std, input_size, (input_mean, input_mean, input_mean), swapRB=True)
	return blob

def preprocess_image_VGG16(image):
	img = cv2.resize(image, (224, 224))
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img_array = np.array(img)
	img_array = np.transpose(img_array, (2, 0, 1))
	img_array = img_array[np.newaxis, :]
	img_array = img_array.astype(np.float32)
	return img_array
	
def crop_and_align_face(image):
	global landmark_predictor
	img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	detector = dlib.get_frontal_face_detector() 
	faces = detector(img_rgb, 1)
	if len(faces) == 0:
		return None  # No face detected
	face = faces[0]
	landmarks = landmark_predictor(img_rgb, face)
	landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
	aligned_face = align_face(img_rgb, landmarks)
	x, y, w, h = face.left(), face.top(), face.width(), face.height()
	cropped_img_rgb = aligned_face[y:y+h, x:x+w]
	return cropped_img_rgb

def align_face(image, landmarks):
	left_eye = np.mean(landmarks[36:42], axis=0)  # Left eye landmarks
	right_eye = np.mean(landmarks[42:48], axis=0)  # Right eye landmarks
	eyes_center = (left_eye + right_eye) / 2.0
	dy = right_eye[1] - left_eye[1]
	dx = right_eye[0] - left_eye[0]
	angle = np.degrees(np.arctan2(dy, dx))
	M = cv2.getRotationMatrix2D(tuple(eyes_center), angle, scale=1)
	aligned_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    
	return aligned_image
	
def detect_morph_differentially(suspected_morph_file_path: str, label: ImageLabel, probe_face_file_path: str) -> Tuple[ReturnCode, bool, float]:
	return_code=ReturnCode.Success
	is_morph=False
	score=0
	try:
		# Load the suspect morph image
		sus_img=cv2.imread(suspected_morph_file_path)
		sus_img=crop_and_align_face(sus_img)
		if sus_img is None:
			return_code = ReturnCode.FaceDetectionError
			return return_code, is_morph, score
		sus_img_array_vgg = preprocess_image_VGG16(sus_img)
		sus_img_array_arc = preprocess_image_arcface(sus_img)
							  

		# Load the probe image
		probe_img=cv2.imread(probe_face_file_path)
		probe_img=crop_and_align_face(probe_img)

		if probe_img is None:
			return_code = ReturnCode.FaceDetectionError
			return return_code, is_morph, score
		probe_img_array_vgg = preprocess_image_VGG16(probe_img)
		probe_img_array_arc = preprocess_image_arcface(probe_img)	
	
		# Detection
		B_model.setInput(sus_img_array_vgg)
		B_vector_sus_img = B_model.forward()
	
		FR_model.setInput(sus_img_array_arc)
		FR_vector_sus_img = FR_model.forward()
	
		B_model.setInput(probe_img_array_vgg)
		B_vector_probe_img = B_model.forward()
	
		FR_model.setInput(probe_img_array_arc)
		FR_vector_probe_img = FR_model.forward()
	
		sus_features = np.concatenate((B_vector_sus_img,FR_vector_sus_img))
		sus_features = sus_features.reshape(1,1024)
		probe_features = np.concatenate((B_vector_probe_img,FR_vector_probe_img))
		probe_features = probe_features.reshape(1,1024)
		diff_vector = sus_features - probe_features
	
		probabilities = SVM_model.predict_proba(diff_vector)
		score = float(probabilities[0,1])
		if score>0.5 : is_morph = True
	except Exception as e:
		return_code=ReturnCode.UnknownError
		return return_code, is_morph, score

	return return_code, is_morph, score

