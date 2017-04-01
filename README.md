# AgeRecognition
School project for recognizing people's age from photographs.
We use a CNN pretrained on AlexNet to guess people age among several classes [0-20, 20-30, 30-40, 40-50, 50-60, 60-70, 70+].

## Computer Vision
Put training images in a "photos" directory.
Use image_face_detection.py to create and save an array of all the images.

## Dataset
The images were collected from the link down below. It consists of 65000 images crawled from Wikipedia (3GB).
https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/