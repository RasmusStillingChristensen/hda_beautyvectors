This repository contains a BOEP submission for morphed image detection using dmad.

The dmad_detect.py contains code to initialize and detect whether an image is morphed by providing a trusted live capture and the suspected morphed image.

The extra_data folder contains 4 different models:

VGG16_512.onnx is a modified VGG16 model trained on [SCUT-FBP5500](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release)

glintr_100.onnx is a ResNET 101 model trained on [Labelled Faces in the Wild (LFW) Dataset](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset) using the [ArcFace](https://insightface.ai/arcface) loss function

shape_predictor_68_face_landmarks.dat is for [dlib](https://pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/) landmarking to align facial images

SVM_FERET_FRGC_MSYNM.pkl is a SVM model trained on DMAD pairs generated using the following morphing methods and datasets:

Morphing methods: 
[UBO-morpher](http://biolab.csr.unibo.it/research.asp?organize=Activities&select=&selObj=220&pathSubj=333%7C%7C22%7C%7C220&Req=&)
[OpenCV](https://learnopencv.com/face-morph-using-opencv-cpp-python/)
[FaceMorpher](https://github.com/yaopang/FaceMorpher/tree/master/facemorpher)
[FaceFusion](http://www.wearemoment.com/FaceFusion)
[MIPGAN](https://arxiv.org/abs/2009.01729)

Image datasets:
[FERET](https://www.nist.gov/itl/products-and-services/color-feret-database)
[FRGC](https://paperswithcode.com/dataset/frgc)
MSYNM


The dmad consists of:

Preprocessing images by detecting faces and landmarks using shape_predictor_68_face_landmarks.dat, to crop and align the trusted live capture and the suspected morph.

Extracting feature and beauty vectors using glintr_100.onnx and VGG16_512.onnx.

Computing a difference vector and calculating a score of the likelyhood of the image being a morph from 0 to 1 with one indicating a morphed image. The score is calculated by the SVM_FERET_FRGC_MSYNM.pkl SVM model.
