# Automatic detection and localization of fiducial markers in CT images

This repository provides the code and materials described in the paper "Automatic fiducial marker detection and localization in volumetric CT images: A combined approach with deep learning"

Parts of the described algorithm are also based on previously reported approaches in [L. Gu and T. Peters (2004)](https://link.springer.com/chapter/10.1007/978-3-540-28626-4_40) and [G. Zheng et al. (2010)](https://pubmed.ncbi.nlm.nih.gov/21096801/).

The design of the spherical fiducial marker available in this repository is protected with the [US patent 10524693](https://patents.justia.com/patent/10524693).

## Fiducial Segmentation

The segmentation code is implemented in C++ and is based on image processing algorithms available in [ITK](https://itk.org/).

[fiducial_segmentation](https://github.com/mregodic/FiducialMarkers/tree/master/fiducial_segmentation) and [fiducial_segmentation_gpu](https://github.com/mregodic/FiducialMarkers/tree/master/fiducial_segmentation_gpu) provide code for running segmentation of volumetric CT images with and without GPU acceleration.

The GPU implementation utilizes [OpenCL accelerated GPU binary morphology image filters for ITK](http://hdl.handle.net/10380/3525).

The [CMake](https://cmake.org/) software can be used to build the project files.

## Fiducial Classification

The deep neural network implemented in Python is trained and used used for multi-fiducial marker classification.

This method also investigates open-set techniques Entropic OpenSet and Objectosphere available in [Reducing Network Agnostophobia](https://github.com/Vastlab/Reducing-Network-Agnostophobia), with certain modifications for general data processing/analysis implemented to their code.

The general structure is the following:

- [fiducial_classification/fiducial_training.py](https://github.com/mregodic/FiducialMarkers/tree/master/fiducial_classification/fiducial_training.py) provides training for the deep neural network.
- [fiducial_classification/fiducial_dataset.py](https://github.com/mregodic/FiducialMarkers/tree/master/fiducial_classification/fiducial_dataset.py) loads the fiducial dataset.
- [fiducial_classification/fiducial_dataset.h5](https://github.com/mregodic/FiducialMarkers/tree/master/fiducial_classification/fiducial_dataset.h5) contains a limited sample of fiducial markers and other structures in order to test the code.
- [fiducial_classification/analyze_3D.py](https://github.com/mregodic/FiducialMarkers/tree/master/fiducial_classification/analyze_3D.py) reads the fiducial datasets, performs predictions using the saved trained models, and analyzes the Open-Set Classification Rate (OSCR) for all four classifiers. 
- [fiducial_classification/evaluation_tools.py](https://github.com/mregodic/FiducialMarkers/tree/master/fiducial_classification/evaluation_tools.py) evaluates the OSCR curve only for a single class and stores the detection rates for thresholds in files for each classifier (see method write_file_best_tresholds).
- [fiducial_classification/evaluate_overall_accuracy.py](https://github.com/mregodic/FiducialMarkers/tree/master/fiducial_classification/evaluate_overall_accuracy.py) evaluates the overall sensitivity, specificity and balanced accuracy based on the determined thresholds (see method write_file_best_tresholds).

The trained models that are used to generate the results in the paper can be obtained in [fiducial_classification/Model_Results/Models](https://github.com/mregodic/FiducialMarkers/fiducial_classification/tree/master/fiducial_classification/Model_Results/Models).

## Fiducial Localization

The fiducial localization code is implemented in C++ using toolkits [ITK](https://itk.org/) and [VTK](https://vtk.org/).

The algorithm creates a mesh of the detected fiducial marker and aligns with the reference mesh using the ICP algorithm.

Once the two meshes are aligned, the fiducial position is calculated based on the determined transformation between the two mesh models.

The [CMake](https://cmake.org/) software can be used to build the project files.

Samples of screw and spherical fiducial data needed to run the code can be found in [fiducial_localization/src/test_data](https://github.com/mregodic/FiducialMarkers/tree/master/fiducial_localization/src/test_data).

## Virtual Phantoms

[virtual_phantoms](https://github.com/mregodic/FiducialMarkers/tree/master/virtual_phantoms) provides mesh models that are used as inputs to [CONRAD](https://www5.cs.fau.de/conrad/) in order to generate a virtual CT used for establishing the ground-truth environment to estimate the localization accuracy.

An example of the setup and how to configure [CONRAD](https://www5.cs.fau.de/conrad/) is described in [virtual_phantoms/virtualct_readme_milo.docx](https://github.com/mregodic/FiducialMarkers/tree/master/virtual_phantoms/virtualct_readme_milo.docx)

The available phantom scenes and mesh models can be opened via [ParaView](https://www.paraview.org/) and [Blender](https://www.blender.org/):

[virtual_phantoms/screw 4p5mm](https://github.com/mregodic/FiducialMarkers/tree/master/virtual_phantoms/screw 4p5mm) - A phantom scene for screws 3.0 mm x 4.5 mm
[virtual_phantoms/screw_3mm](https://github.com/mregodic/FiducialMarkers/tree/master/virtual_phantoms/screw_3mm) - A phantom scene for screws 2.0 mm x 3.0 mm
[virtual_phantoms/screw_3p75mm](https://github.com/mregodic/FiducialMarkers/tree/master/virtual_phantoms/screw_3p75mm) - A phantom scene for screws 3.0 mm x 3.75 mm
[virtual_phantoms/spherical_3mm](https://github.com/mregodic/FiducialMarkers/tree/master/virtual_phantoms/spherical_3mm) - A phantom scene for spherical fiducial 3.0 mm x 6.0 mm
[virtual_phantoms/spherical_4mm](https://github.com/mregodic/FiducialMarkers/tree/master/virtual_phantoms/spherical_4mm) - A phantom scene for spherical fiducial 4.0 mm x 8.0 mm