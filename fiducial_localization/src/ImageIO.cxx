
#include "ImageIO.h"

ImageIO::ImageType::Pointer ImageIO::ReadImageFile(std::string fileName)
{
	try
	{
		std::cout << "Read image from " << fileName << std::endl;

		typedef itk::ImageFileReader<ImageType> ReaderType;

		ReaderType::Pointer reader = ReaderType::New();

		reader->SetFileName(fileName);
		reader->Update();

		return reader->GetOutput();
	}
	catch (itk::ExceptionObject & err)
	{
		std::cerr << "ExceptionObject caught !" << std::endl;
		std::cerr << err << std::endl;
	}
}

void ImageIO::WriteImageFile(ImageType::Pointer transformedImage, std::string fileName)
{
	try
	{
		std::cout << "Saving image to " << fileName << std::endl;

		typedef itk::GDCMImageIO ImageIOType;
		typedef itk::ImageFileWriter<ImageType> WriterType;

		ImageIOType::Pointer imageIO = ImageIOType::New();

		WriterType::Pointer writer = WriterType::New();

		writer->SetFileName(fileName);
		writer->SetInput(transformedImage);
		writer->SetImageIO(imageIO);

		writer->Update();
	}
	catch (itk::ExceptionObject & err)
	{
		std::cerr << "ExceptionObject caught !" << std::endl;
		std::cerr << err << std::endl;
	}
}

ImageIO::BinaryImageType::Pointer ImageIO::BinarizeImage(ImageType::Pointer image, int lowerThreshold, int upperThreshold)
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

ImageIO::ImageType::Pointer ImageIO::Threshold(ImageType::Pointer image, int lowerThreshold, int upperThreshold)
{
	std::cout << "binary threshold started" << std::endl;

	typedef itk::ThresholdImageFilter<ImageType> ThresholdImageFilterType;

	ThresholdImageFilterType::Pointer thresholdFilter = ThresholdImageFilterType::New();
	thresholdFilter->SetInput(image);
	thresholdFilter->SetLower(lowerThreshold);
	thresholdFilter->SetUpper(upperThreshold);
	thresholdFilter->SetOutsideValue(-1000);

	thresholdFilter->Update();

	std::cout << "binary threshold ended" << std::endl;

	return thresholdFilter->GetOutput();
}
