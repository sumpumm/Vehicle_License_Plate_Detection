
characterdet - v2 2023-08-09 9:13pm
==============================

This dataset was exported via roboflow.com on December 5, 2023 at 12:23 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 936 images.
Number-plate-detection are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)

The following augmentation was applied to create 3 versions of each source image:
* Randomly crop between 0 and 35 percent of the image
* Random rotation of between -21 and +21 degrees
* Random shear of between -19° to +19° horizontally and -11° to +11° vertically
* Random brigthness adjustment of between -25 and +25 percent
* Random exposure adjustment of between -25 and +25 percent


