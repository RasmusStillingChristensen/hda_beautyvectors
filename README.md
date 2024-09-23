This repository contains a BOEP submission for morphed image detection using dmad.

This dmad method uses feature vectors extracted using a ResNET 101 model trained on [Labelled Faces in the Wild (LFW) Dataset](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset) using the [ArcFace](https://insightface.ai/arcface) loss function together with beauty feature vectors extracted using a modified VGG16 model trained on [SCUT-FBP5500](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release).

Morphs generated using the morphing methods: 
[UBO-morpher](http://biolab.csr.unibo.it/research.asp?organize=Activities&select=&selObj=220&pathSubj=333%7C%7C22%7C%7C220&Req=&)
[OpenCV](https://learnopencv.com/face-morph-using-opencv-cpp-python/)
[FaceMorpher](https://github.com/yaopang/FaceMorpher/tree/master/facemorpher)
[FaceFusion](http://www.wearemoment.com/FaceFusion)
[MIPGAN](https://arxiv.org/abs/2009.01729)

Images morphed using the morphing methods come from:
[FERET](https://www.nist.gov/itl/products-and-services/color-feret-database)
[FRGC](https://paperswithcode.com/dataset/frgc)
MSYNM


Image preprocessing consists of loading the dmad image pair, using [dlib](https://pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/) to detect and align faces in the image based on landmarks.

Feature vectors and beauty vectors are extracted from the preprocessed images using the ResNET 101 model and VGG16 model. A difference vector is calculated by concatentating the feature and beauty vector and then substracting calculating the difference between the images using substraction.

The difference vectors are classified using an SVM model trained difference vectors from dmad pairs generated using the morphing methods on FERET, FRGC and MSYNM datasets.

