#ifndef __ImageIO_h
#define __ImageIO_h

#include "itkImage.h"

#include "itkGDCMImageIO.h"
#include "itkImageSeriesReader.h"
#include "itkImageFileWriter.h"
#include "itkThresholdImageFilter.h"

#include "itkEuler3DTransform.h"
#include "itkResampleImageFilter.h"
#include "itkBinaryThresholdImageFilter.h"

#include "itkVTKPolyDataWriter.h"

class ImageIO
{

public:
	typedef unsigned char BinaryPixelType;

	typedef itk::Image<BinaryPixelType, 3>  BinaryImageType;

	typedef float PixelType;

	typedef itk::Image<PixelType, 3>  ImageType;

	typedef itk::Euler3DTransform<double> TransformType;

public:
	static ImageType::Pointer ReadImageFile(std::string fileName);

	static void WriteImageFile(ImageType::Pointer transformedImage, std::string fileName);

	static BinaryImageType::Pointer BinarizeImage(ImageType::Pointer image, int lowerThreshold, int upperThreshold);

	static ImageIO::ImageType::Pointer Threshold(ImageType::Pointer image, int lowerThreshold, int upperThreshold);

protected:
	ImageIO() {}
	~ImageIO() {}

};

#endif