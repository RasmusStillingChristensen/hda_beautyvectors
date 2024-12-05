import os
import numpy as np
import cv2
import onnxruntime as ort
import pickle
import dlib
from typing import Tuple
from enum import IntEnum
from sklearn import svm


# Global variables
B_model = None
FR_model = None
SVM_model = None
landmark_predictor = None
face_crop_model = None

class ReturnCode(IntEnum):
	Success = 0
	UnknownError = 1
	InputLocationError = 2
	FaceDetectionError = 3


class ImageLabel(IntEnum):
	Unknown = 0
	NonScanned = 1
	Scanned = 2


def initialize(alg_extra_data_folder_path: str) -> ReturnCode:
	return_code = ReturnCode.Success
	global B_model, FR_model, SVM_model, landmark_predictor, face_crop_model

	try:
		# Construct paths to the models
		b_model_path = os.path.join(alg_extra_data_folder_path, "VGG16_512.onnx")
		fr_model_path = os.path.join(alg_extra_data_folder_path, "glintr100.onnx")
		svm_model_path = os.path.join(alg_extra_data_folder_path, "SVM_FERET_FRGC_MSYNM_v2.pkl")
		
		landmark_predictor_path = os.path.join(alg_extra_data_folder_path, "shape_predictor_68_face_landmarks.dat")
		face_crop_model_path = os.path.join(alg_extra_data_folder_path, "res10_300x300_ssd_iter_140000_fp16.caffemodel")
		face_crop_config_path = os.path.join(alg_extra_data_folder_path, "deploy.prototxt")
		
		# Load ONNX models using onnxruntime
		providers = ['CUDAExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
		B_model = ort.InferenceSession(b_model_path)
		FR_model = ort.InferenceSession(fr_model_path)

		# Load SVM model using pickle
		with open(svm_model_path, 'rb') as file:
			SVM_model = pickle.load(file)
		
		# Load dlib landmark predictor
		landmark_predictor = dlib.shape_predictor(landmark_predictor_path)
		
		# Load face crop model
		face_crop_model = cv2.dnn.readNetFromCaffe(face_crop_config_path, face_crop_model_path)
		
	except Exception as e:
		print(f"Error initializing models: {e}")
		return_code = ReturnCode.InputLocationError

	return return_code

preferred_providers = [
	'CUDAExecutionProvider',
	'TensorRTExecutionProvider',
	'DirectMLExecutionProvider',
	'OpenVINOExecutionProvider',
	'CoreMLExecutionProvider',
	'CPUExecutionProvider' 

def get_best_provider():
	available_providers = ort.get_available_providers()
    
	for provider in preferred_providers:
		if provider in available_providers:
			return provider
    
	return 'CPUExecutionProvider'

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
	img_array = np.transpose(img_array, (2, 0, 1))  # Convert to (C, H, W)
	img_array = img_array[np.newaxis, :]  # Add batch dimension
	img_array = img_array.astype(np.float32)
	return img_array


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


def crop_face(image):
	global face_crop_model
	net = face_crop_model
	h, w = image.shape[:2]
	blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104.0, 177.0, 123.0], False, False)
	net.setInput(blob)
	detections = net.forward()
	for i in range(detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		if confidence > 0.5:  # Threshold to filter out weak detections
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(x1, y1, x2, y2) = box.astype("int")
			x1, y1 = max(0, x1), max(0, y1)
			x2, y2 = min(w, x2), min(h, y2)
			cropped_img = image[y1:y2, x1:x2]

			return cropped_img

	return None  # No face detected


def detect_morph_differentially(suspected_morph_file_path: str, label: ImageLabel, probe_face_file_path: str) -> Tuple[ReturnCode, bool, float]:
	return_code = ReturnCode.Success
	is_morph = False
	score = 0
	try:
		# Load the suspect morph image
		sus_img = cv2.imread(suspected_morph_file_path)

		if sus_img is None:
			return_code = ReturnCode.InputLocationError
			return return_code, is_morph, score
		sus_img = crop_face(sus_img)

		if sus_img is None:
			return_code = ReturnCode.FaceDetectionError
			return return_code, is_morph, score
		sus_img_array_vgg = preprocess_image_VGG16(sus_img)
		sus_img_array_arc = preprocess_image_arcface(sus_img)
		
		# Load the probe image
		probe_img = cv2.imread(probe_face_file_path)

		if probe_img is None:
			return_code = ReturnCode.InputLocationError
			return return_code, is_morph, score
		probe_img = crop_face(probe_img)

		if probe_img is None:
			return_code = ReturnCode.FaceDetectionError
			return return_code, is_morph, score
		probe_img_array_vgg = preprocess_image_VGG16(probe_img)
		probe_img_array_arc = preprocess_image_arcface(probe_img)	

		# Run the onnx models for the suspect image
		B_vector_sus_img = B_model.run(None, {"input.1": sus_img_array_vgg.astype(np.float32)})[0]
		FR_vector_sus_img = FR_model.run(None, {"input.1": sus_img_array_arc.astype(np.float32)})[0]

		# Run the onnx models for the probe image
		B_vector_probe_img = B_model.run(None, {"input.1": probe_img_array_vgg.astype(np.float32)})[0]
		FR_vector_probe_img = FR_model.run(None, {"input.1": probe_img_array_arc.astype(np.float32)})[0]

		# Concatenate feature vectors
		sus_features = np.concatenate((B_vector_sus_img, FR_vector_sus_img))
		sus_features = sus_features.reshape(1, 1024)
		probe_features = np.concatenate((B_vector_probe_img, FR_vector_probe_img))
		probe_features = probe_features.reshape(1, 1024)

		diff_vector = sus_features - probe_features

		# SVM prediction
		probabilities = SVM_model.predict_proba(diff_vector)
		score = float(probabilities[0, 1])
		if score > 0.5:
			is_morph = True

	except Exception as e:
		return_code = ReturnCode.UnknownError
		return return_code, is_morph, score

	return return_code, is_morph, score

