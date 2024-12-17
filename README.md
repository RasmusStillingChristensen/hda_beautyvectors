# BOEP Submission: Morphed Image Detection Using DMAD

This repository contains a submission for morphed image detection utilizing **DMAD** (Difference-based Morph Anomaly Detection).

## Overview of the Algorithm
The DMAD algorithm detects morphed images by comparing a trusted live capture with a suspected morph. It preprocesses the images, extracts feature vectors, computes differences, and scores the likelihood of morphing using a trained SVM model.

---

## File Structure

### **`dmad_detect.py`**
This script initializes the detection process, processes input images, and outputs a morph likelihood score (0 to 1), where 1 indicates a morphed image.

### **`extra_data/`**
This folder contains pre-trained models and supporting files essential for D-MAD:

- **`VGG16_512.onnx`**: A modified VGG16 model trained on [SCUT-FBP5500](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release) for extracting feature vectors based on facial attractiveness.
- **`glintr_100.onnx`**: A ResNet 101 model trained on [LFW Dataset](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset) using the [ArcFace](https://insightface.ai/arcface) loss function for facial recognition.
- **`shape_predictor_68_face_landmarks.dat`**: A facial landmarking model for image alignment using [dlib](https://pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/).
- **`res10_300x300_ssd_iter_140000_fp16.caffemodel` and `deploy.prototxt`**: Files used for face detection and cropping via OpenCV's DNN module.
- **`SVM_FERET_FRGC_MSYNM_v2.pkl`**: An SVM model trained on feature differences from multiple datasets and morphing methods.

---

## Training Details

#### **Morphing Methods**
- [UBO-morpher](http://biolab.csr.unibo.it/research.asp?organize=Activities&select=&selObj=220&pathSubj=333%7C%7C22%7C%7C220&Req=&)
- [OpenCV](https://learnopencv.com/face-morph-using-opencv-cpp-python/)
- [FaceMorpher](https://github.com/yaopang/FaceMorpher/tree/master/facemorpher)
- [FaceFusion](http://www.wearemoment.com/FaceFusion)
- [MIPGAN](https://arxiv.org/abs/2009.01729)

#### **Datasets Used**
- [FERET](https://www.nist.gov/itl/products-and-services/color-feret-database)
- [FRGC](https://paperswithcode.com/dataset/frgc)
- **MSYNM**

---

## D-MAD Workflow

1. **Initialization**:  
   Load the required pre-trained models, including CNN models, SVM, face detectors, and landmark predictors.

2. **Preprocessing**:  
   - Detect and crop faces using OpenCV's DNN model.
   - Align faces based on dlib landmarks to standardize orientation.

3. **Feature Extraction**:  
   - Extract beauty-related feature vectors using `VGG16_512.onnx`.
   - Extract facial recognition feature vectors using `glintr_100.onnx`.

4. **Difference Vector Calculation**:  
   Compute the difference between the trusted and suspected feature vectors.

5. **Classification**:  
   Use the SVM model (`SVM_FERET_FRGC_MSYNM_v2.pkl`) to classify the difference vectors and score the likelihood of the suspected image being morphed.

---

## How to Use
### **Initialization**
Before running the algorithm, initialize the models:
```python
initialize("path_to_extra_data")
```

### **Detection**
Run the detection function to calculate the morph likelihood:
```python
detect_morph_differentially(
    suspected_morph_file_path="path_to_suspected_image.jpg",
    label=ImageLabel.Unknown,
    probe_face_file_path="path_to_trusted_image.jpg"
)
```

The function returns:
- `ReturnCode`: Success or error code.
- `is_morph`: Boolean indicating whether the image is likely morphed.
- `score`: Likelihood score (0 to 1).

---

## Dependencies
- Python 3.x
- OpenCV
- ONNXRuntime
- dlib
- NumPy
- scikit-learn