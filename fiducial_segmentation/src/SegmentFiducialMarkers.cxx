
#include <iostream>
#include <string>

#include "itkImage.h"
#include "itkImageSeriesReader.h"
#include "itkImageFileWriter.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkMaskImageFilter.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkThresholdImageFilter.h"
#include "itkBinaryBallStructuringElement.h"
#include "itkBinaryErodeImageFilter.h"
#include "itkBinaryDilateImageFilter.h"
#include "itkMedianImageFilter.h"
#include "itkAndImageFilter.h"
#include "itkTestingComparisonImageFilter.h"

typedef float PixelType;
typedef itk::Image<PixelType, 3>  ImageType;
typedef itk::Image<unsigned char, 3>  BinaryImageType;

typedef itk::BinaryBallStructuringElement<BinaryImageType::PixelType, BinaryImageType::ImageDimension> StructuringElementType;

ImageType::Pointer ReadSeriesImage(std::string folderName);
BinaryImageType::Pointer ApplyBinaryThreshold(ImageType::Pointer image, int lowerThreshold, int upperThreshold);
ImageType::Pointer ApplyGrayscaleThreshold(ImageType::Pointer image, int lowerThreshold, int upperThreshold);
BinaryImageType::Pointer BinaryDilation(BinaryImageType::Pointer image, const unsigned int radius);
BinaryImageType::Pointer BinaryErosion(BinaryImageType::Pointer image, const unsigned int radius);
ImageType::Pointer MedianImageFilter(ImageType::Pointer image, unsigned int radius);
BinaryImageType::Pointer ConditionalDilation(BinaryImageType::Pointer image, BinaryImageType::Pointer maskImage);
ImageType::Pointer RecoverIntensityImage(ImageType::Pointer image, BinaryImageType::Pointer imageMask);
void WriteImageFile(ImageType::Pointer image, std::string fileName);

int main()
{
	bool medianNoiseReduction = false;

	const std::string imagePath = "F:\\dicomdir";
	const std::string outputImageFile = "F:\\output.mhd";

	ImageType::Pointer image;
	image = ReadSeriesImage(imagePath);

	BinaryImageType::Pointer binaryImage;

	if (medianNoiseReduction)
	{
		ImageType::Pointer tmpImage;
		tmpImage = ApplyGrayscaleThreshold(image, 2500, 100000);

		// median sequence
		for (int i = 0; i < 1; i++)
		{
			tmpImage = MedianImageFilter(tmpImage, 2);
		}

		// binarized image
		binaryImage = ApplyBinaryThreshold(tmpImage, 2500, 100000);
	}
	else
	{
		// binarized image
		binaryImage = ApplyBinaryThreshold(image, 2500, 100000);

		// binary opening sequence

		// dilation
		binaryImage = BinaryDilation(binaryImage, 1);

		// erosion sequence
		for (int i = 0; i < 2; i++)
		{
			binaryImage = BinaryErosion(binaryImage, 1);
		}
	}

	BinaryImageType::Pointer maskImage;
	maskImage = ApplyBinaryThreshold(image, 2000, 100000);

	binaryImage = ConditionalDilation(binaryImage, maskImage);

	ImageType::Pointer outputImage;
	outputImage = RecoverIntensityImage(image, binaryImage);

	WriteImageFile(outputImage, outputImageFile);

	int a;
	std::cin >> a;
}

ImageType::Pointer ReadSeriesImage(std::string folderName)
{
	typedef itk::GDCMImageIO ImageIOType;
	typedef itk::GDCMSeriesFileNames NamesGeneratorType;
	typedef itk::ImageSeriesReader<ImageType> ReaderType;

	std::cout << "Reading: " << folderName << std::endl;

	ImageIOType::Pointer imageIO = ImageIOType::New();

	NamesGeneratorType::Pointer m_NameGenerator = NamesGeneratorType::New();
	m_NameGenerator->SetInputDirectory(folderName);
	itk::FilenamesContainer fileNames1 = m_NameGenerator->GetInputFileNames();

	ReaderType::Pointer reader = ReaderType::New();
	reader->SetImageIO(imageIO);
	reader->SetFileNames(fileNames1);
	reader->Update();

	return reader->GetOutput();
}


void WriteImageFile(ImageType::Pointer image, std::string fileName)
{
	try
	{
		std::cout << "Saving image to " << fileName << std::endl;

		typedef itk::ImageFileWriter<ImageType> WriterType;

		WriterType::Pointer writer = WriterType::New();

		writer->SetFileName(fileName);
		writer->SetInput(image);

		writer->Update();
	}
	catch (itk::ExceptionObject & err)
	{
		std::cerr << "ExceptionObject caught !" << std::endl;
		std::cerr << err << std::endl;
	}
}


BinaryImageType::Pointer ApplyBinaryThreshold(ImageType::Pointer image, int lowerThreshold, int upperThreshold)
{
	std::cout << "binary threshold started" << std::endl;

	typedef itk::BinaryThresholdImageFilter<ImageType, BinaryImageType> BinaryThresholdImageFilterType;

	BinaryThresholdImageFilterType::Pointer thresholdFilter = BinaryThresholdImageFilterType::New();
	thresholdFilter->SetInput(image);
	thresholdFilter->SetLowerThreshold(lowerThreshold);
	thresholdFilter->SetUpperThreshold(upperThreshold);
	thresholdFilter->SetInsideValue(255);
	thresholdFilter->SetOutsideValue(0);

	thresholdFilter->Update();

	std::cout << "binary threshold ended" << std::endl;

	return thresholdFilter->GetOutput();
}

ImageType::Pointer ApplyGrayscaleThreshold(ImageType::Pointer image, int lowerThreshold, int upperThreshold)
{
	std::cout << "grayscale threshold started" << std::endl;

	typedef itk::ThresholdImageFilter<ImageType> ThresholdImageFilterType;

	ThresholdImageFilterType::Pointer thresholdFilter = ThresholdImageFilterType::New();
	thresholdFilter->SetInput(image);
	thresholdFilter->ThresholdOutside(lowerThreshold, upperThreshold);
	thresholdFilter->SetOutsideValue(0);
	thresholdFilter->Update();

	std::cout << "grayscale threshold ended" << std::endl;

	return thresholdFilter->GetOutput();
}

StructuringElementType StructuringElement(unsigned char radius)
{
	std::cout << "Radius: " << radius << std::endl;
	StructuringElementType::SizeType size;

	size[0] = 0;
	size[1] = radius;
	size[2] = radius;

	StructuringElementType structuringElement;
	structuringElement.SetRadius(size);
	structuringElement.CreateStructuringElement();
	//structuringElement.Print(std::cout);

	return structuringElement;
}

BinaryImageType::Pointer BinaryDilation(BinaryImageType::Pointer image, const unsigned int radius)
{
	std::cout << "Binary Dilation started " << std::endl;
	
	typedef itk::BinaryDilateImageFilter <BinaryImageType, BinaryImageType, StructuringElementType> BinaryDilateImageFilter;

	StructuringElementType structuringElement = StructuringElement(radius);

	BinaryDilateImageFilter::Pointer dilateFilter = BinaryDilateImageFilter::New();
	dilateFilter->SetInput(image);
	dilateFilter->SetForegroundValue(255);
	dilateFilter->SetBackgroundValue(0);
	dilateFilter->SetKernel(structuringElement);
	dilateFilter->Update();

	std::cout << "Binary Dilation ended " << std::endl;

	return dilateFilter->GetOutput();

}

BinaryImageType::Pointer BinaryErosion(BinaryImageType::Pointer image, const unsigned int radius)
{
	std::cout << "Binary Erosion started" << std::endl;
	
	typedef itk::BinaryErodeImageFilter <BinaryImageType, BinaryImageType, StructuringElementType> BinaryErodeImageFilter;

	StructuringElementType structuringElement = StructuringElement(radius);

	BinaryErodeImageFilter::Pointer erodeFilter = BinaryErodeImageFilter::New();
	erodeFilter->SetInput(image);
	erodeFilter->SetForegroundValue(255);
	erodeFilter->SetBackgroundValue(0);
	erodeFilter->SetKernel(structuringElement);
	erodeFilter->Update();

	std::cout << "Binary Erosion ended" << std::endl;

	return erodeFilter->GetOutput();
}

ImageType::Pointer MedianImageFilter(ImageType::Pointer image, unsigned int radius)
{
	std::cout << std::endl << "Median Filter started" << std::endl;

	using MedianImageFilterType = itk::MedianImageFilter< ImageType, ImageType >;
	MedianImageFilterType::Pointer medianImageFilter = MedianImageFilterType::New();
	medianImageFilter->SetInput(image);
	medianImageFilter->SetRadius(radius);
	medianImageFilter->Update();

	std::cout << std::endl << "Median Filter ended" << std::endl;

	return medianImageFilter->GetOutput();
}

BinaryImageType::Pointer ConditionalDilation(BinaryImageType::Pointer binaryImage, BinaryImageType::Pointer maskImage)
{
	BinaryImageType::Pointer testImage;

	int i = 0;
	int previousNumberOfDifferences = -1;

	// stop conditions
	const int maximumIterations = 20;
	const int maximumVoxelDifference = 5;

	while (i++ <= maximumIterations)
	{
		std::cout << std::endl << "Conditional dilation iteratation " << i << std::endl;

		testImage = binaryImage;

		// Dilate Image
		binaryImage = BinaryDilation(binaryImage, 1);

		// Restrict within the defined region of interest
		typedef itk::AndImageFilter<BinaryImageType> AndImageFilterType;
		AndImageFilterType::Pointer andFilter = AndImageFilterType::New();
		andFilter->SetInput(0, binaryImage);
		andFilter->SetInput(1, maskImage);
		andFilter->Update();

		binaryImage = andFilter->GetOutput();

		// Get number of differences between the previous and current dilated image.
		typedef itk::Testing::ComparisonImageFilter<BinaryImageType, BinaryImageType> ComparisonImageFilterType;
		ComparisonImageFilterType::Pointer diff = ComparisonImageFilterType::New();
		diff->SetValidInput(binaryImage);
		diff->SetTestInput(testImage);
		diff->UpdateLargestPossibleRegion();

		const unsigned long numberOfDifferences = diff->GetNumberOfPixelsWithDifferences();

		std::cout << "Number of pixels with differences " << numberOfDifferences << std::endl;

		// Stop condition 2
		if (numberOfDifferences <= maximumVoxelDifference)
		{
			// if the images are the same
			break;
		}

		// Stop condition 3
		if (i > 2 && numberOfDifferences >= previousNumberOfDifferences)
		{
			std::cout << "at i = " << i << " difference different than previous, break -- current: " << numberOfDifferences << ", previous: " << previousNumberOfDifferences << std::endl;
			// This means that some other elements have started to connect to the current segmented element
			// We don't want that, and we use the previous image
			binaryImage = testImage;
			--i; // reducing conditional dilation sum for 1
			break;
		}
	}

	return binaryImage;
}


ImageType::Pointer RecoverIntensityImage(ImageType::Pointer image, BinaryImageType::Pointer imageMask)
{
	std::cout << "Mask started" << std::endl;

	typedef itk::MaskImageFilter<ImageType, BinaryImageType> MaskFilterType;
	MaskFilterType::Pointer maskFilter = MaskFilterType::New();
	maskFilter->SetInput(image);
	maskFilter->SetMaskImage(imageMask);
	maskFilter->Update();

	std::cout << "Mask done" << std::endl;

	return maskFilter->GetOutput();
}