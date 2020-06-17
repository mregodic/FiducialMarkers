
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

#include "itkGPUBinaryThresholdImageFilter.h"
#include "itkGPUBinaryErodeImageFilter.h"
#include "itkGPUBinaryDilateImageFilter.h"
#include "itkRegionOfInterestImageFilter.h"

typedef float PixelType;
typedef itk::Image<PixelType, 3>  ImageType;

typedef itk::ImageRegion< 3 > ImageRegionType;
typedef itk::GPUImage<unsigned int, 3> GPUOutImageType;
typedef itk::GPUImage<int, 3> GPUInImageType;

typedef itk::BinaryBallStructuringElement<GPUOutImageType::PixelType, GPUOutImageType::ImageDimension> StructuringElementType;

GPUInImageType::Pointer SelectRegionAndConvertToGPUImage(ImageType::Pointer image);
ImageType::Pointer ReadSeriesImage(std::string folderName);
GPUOutImageType::Pointer ApplyBinaryThreshold(GPUInImageType::Pointer image, int lowerThreshold, int upperThreshold);
ImageType::Pointer ApplyGrayscaleThreshold(ImageType::Pointer image, int lowerThreshold, int upperThreshold);
GPUOutImageType::Pointer BinaryDilation(GPUOutImageType::Pointer image, const unsigned int radius);
GPUOutImageType::Pointer BinaryErosion(GPUOutImageType::Pointer image, const unsigned int radius);
ImageType::Pointer MedianImageFilter(ImageType::Pointer image, unsigned int radius);
GPUOutImageType::Pointer ConditionalDilation(GPUOutImageType::Pointer image, GPUOutImageType::Pointer maskImage);
ImageType::Pointer RecoverIntensityImage(ImageType::Pointer image, GPUOutImageType::Pointer imageMask);
void WriteImageFile(ImageType::Pointer image, std::string fileName);

int main()
{
	bool medianNoiseReduction = false;

	const std::string imagePath = "F:\\dicomdir";
	const std::string outputImageFile = "F:\\output.mhd";

	ImageType::Pointer image;
	image = ReadSeriesImage(imagePath);

	GPUInImageType::Pointer gpuImage;
	gpuImage = SelectRegionAndConvertToGPUImage(image);

	GPUOutImageType::Pointer gpuBinaryImage;

	if (medianNoiseReduction)
	{
		ImageType::Pointer tmpImage;
		tmpImage = ApplyGrayscaleThreshold(image, 2500, 100000);

		// median sequence
		for (int i = 0; i < 1; i++)
		{
			tmpImage = MedianImageFilter(tmpImage, 2);
		}

		GPUInImageType::Pointer gpuImageFiltered;
		gpuImageFiltered = SelectRegionAndConvertToGPUImage(tmpImage);

		// binarized image
		gpuBinaryImage = ApplyBinaryThreshold(gpuImageFiltered, 2500, 100000);
	}
	else
	{
		
		// binarized image
		gpuBinaryImage = ApplyBinaryThreshold(gpuImage, 2500, 100000);

		// binary opening sequence

		// dilation
		gpuBinaryImage = BinaryDilation(gpuBinaryImage, 1);

		// erosion sequence
		for (int i = 0; i < 2; i++)
		{
			gpuBinaryImage = BinaryErosion(gpuBinaryImage, 1);
		}
	}

	GPUOutImageType::Pointer maskImage;
	maskImage = ApplyBinaryThreshold(gpuImage, 2000, 100000);

	gpuBinaryImage = ConditionalDilation(gpuBinaryImage, maskImage);

	ImageType::Pointer outputImage;
	outputImage = RecoverIntensityImage(image, gpuBinaryImage);

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


GPUInImageType::Pointer SelectRegionAndConvertToGPUImage(ImageType::Pointer image)
{
	// ROI update
	typedef itk::RegionOfInterestImageFilter<ImageType, GPUInImageType > ROIFilterType;
	
	ImageRegionType desiredRegion = image->GetLargestPossibleRegion();

	ROIFilterType::Pointer roiFilter = ROIFilterType::New();
	roiFilter->SetRegionOfInterest(desiredRegion);
	roiFilter->SetInput(image);
	roiFilter->Update();
	std::cout << "ROI update done" << std::endl;

	return roiFilter->GetOutput();
}

GPUOutImageType::Pointer ApplyBinaryThreshold(GPUInImageType::Pointer image, int lowerThreshold, int upperThreshold)
{
	
	std::cout << "binary threshold started" << std::endl;

	typedef itk::GPUBinaryThresholdImageFilter<GPUInImageType, GPUOutImageType> GPUBinaryThresholdImageFilterType;

	GPUBinaryThresholdImageFilterType::Pointer thresholdFilter = GPUBinaryThresholdImageFilterType::New();
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

GPUOutImageType::Pointer BinaryDilation(GPUOutImageType::Pointer image, const unsigned int radius)
{
	std::cout << "Binary Dilation started " << std::endl;
	
	typedef itk::BinaryDilateImageFilter <GPUOutImageType, GPUOutImageType, StructuringElementType> BinaryDilateImageFilter;

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

GPUOutImageType::Pointer BinaryErosion(GPUOutImageType::Pointer image, const unsigned int radius)
{
	std::cout << "Binary Erosion started" << std::endl;
	
	typedef itk::BinaryErodeImageFilter <GPUOutImageType, GPUOutImageType, StructuringElementType> BinaryErodeImageFilter;

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

GPUOutImageType::Pointer ConditionalDilation(GPUOutImageType::Pointer binaryImage, GPUOutImageType::Pointer maskImage)
{
	GPUOutImageType::Pointer testImage;

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
		typedef itk::AndImageFilter<GPUOutImageType> AndImageFilterType;
		AndImageFilterType::Pointer andFilter = AndImageFilterType::New();
		andFilter->SetInput(0, binaryImage);
		andFilter->SetInput(1, maskImage);
		andFilter->Update();

		binaryImage = andFilter->GetOutput();

		// Get number of differences between the previous and current dilated image.
		typedef itk::Testing::ComparisonImageFilter<GPUOutImageType, GPUOutImageType> ComparisonImageFilterType;
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


ImageType::Pointer RecoverIntensityImage(ImageType::Pointer image, GPUOutImageType::Pointer imageMask)
{
	std::cout << "Recovering intensity started" << std::endl;

	typedef itk::Image<unsigned char, 3>  BinaryImageType;
	
	typedef itk::CastImageFilter<GPUOutImageType, BinaryImageType> CastFilterType;
	CastFilterType::Pointer castFilter = CastFilterType::New();
	castFilter->SetInput(imageMask);
	castFilter->Update();
	std::cout << "Cast filter done" << std::endl;

	typedef itk::MaskImageFilter<ImageType, BinaryImageType> MaskFilterType;
	MaskFilterType::Pointer maskFilter = MaskFilterType::New();
	maskFilter->SetInput(image);
	maskFilter->SetMaskImage(castFilter->GetOutput());
	maskFilter->Update();

	std::cout << "Recovering intensity ended" << std::endl;

	return maskFilter->GetOutput();
}