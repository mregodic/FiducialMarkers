
#include "MeshExtraction.h"

#include <vtkPolyDataConnectivityFilter.h>
#include <vtkImageResize.h>
#include <vtkWindowedSincPolyDataFilter.h>

MeshExtraction::MeshType::Pointer MeshExtraction::ReadMeshFile(std::string fileName)
{
	try
	{
		itk::STLMeshIOFactory::RegisterOneFactory();

		std::cout << "Read mesh from " << fileName << std::endl;

		typedef itk::MeshFileReader<MeshType> ReaderType;

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

vtkSmartPointer<vtkImageData> MeshExtraction::ConvertItkToVtkImage(ImageType::Pointer image)
{
	typedef itk::ImageToVTKImageFilter<ImageType> itkVtkConverter;
	itkVtkConverter::Pointer converter = itkVtkConverter::New();
	converter->SetInput(image);
	converter->Update();

	vtkSmartPointer<vtkImageData> vtkImage = vtkSmartPointer<vtkImageData>::New();

	vtkImage->DeepCopy(converter->GetOutput());

	ImageType::PointType origin = image->GetOrigin();
	ImageType::SpacingType spacing = image->GetSpacing();
	vtkImage->SetOrigin(origin[0], origin[1], origin[2]);
	vtkImage->SetSpacing(spacing[0], spacing[1], spacing[2]);

	return vtkImage;
}

vtkSmartPointer<vtkImageData> MeshExtraction::ConvertItkToVtkImageBinary(BinaryImageType::Pointer image)
{
	typedef itk::ImageToVTKImageFilter<BinaryImageType> itkVtkConverter;
	itkVtkConverter::Pointer converter = itkVtkConverter::New();
	converter->SetInput(image);
	converter->Update();

	vtkSmartPointer<vtkImageData> vtkImage = vtkSmartPointer<vtkImageData>::New();

	vtkImage->DeepCopy(converter->GetOutput());

	BinaryImageType::PointType origin = image->GetOrigin();
	BinaryImageType::SpacingType spacing = image->GetSpacing();
	vtkImage->SetOrigin(origin[0], origin[1], origin[2]);
	vtkImage->SetSpacing(spacing[0], spacing[1], spacing[2]);

	return vtkImage;
}

vtkSmartPointer<vtkPolyData> MeshExtraction::CreateMeshBinary(BinaryImageType::Pointer image)
{
	double isoValue = 255;

	vtkSmartPointer<vtkImageData> vtkImageData;
	vtkImageData = ConvertItkToVtkImageBinary(image);

	double fractionalOversamplingFactor = 1;

	// Resize the image with interpolation, this helps the conversion for structures with small labelmaps
	vtkSmartPointer<vtkImageResize> imageResize = vtkSmartPointer<vtkImageResize>::New();
	imageResize->SetInputData(vtkImageData);
	imageResize->BorderOn();
	imageResize->SetResizeMethodToMagnificationFactors();
	imageResize->SetMagnificationFactors(fractionalOversamplingFactor, fractionalOversamplingFactor, fractionalOversamplingFactor);
	imageResize->InterpolateOn();
	imageResize->Update();
	vtkImageData = imageResize->GetOutput();

	vtkSmartPointer<vtkFlyingEdges3D> marchingCubes = vtkSmartPointer<vtkFlyingEdges3D>::New();
	//vtkSmartPointer<vtkMarchingCubes> marchingCubes = vtkSmartPointer<vtkMarchingCubes>::New();
	marchingCubes->SetInputData(vtkImageData);
	marchingCubes->SetNumberOfContours(1);
	marchingCubes->SetValue(0, isoValue);
	marchingCubes->ComputeScalarsOff();
	marchingCubes->ComputeGradientsOff();
	marchingCubes->ComputeNormalsOff();

	marchingCubes->Update();

	vtkSmartPointer<vtkPolyData> processingResult = marchingCubes->GetOutput();

	if (processingResult->GetNumberOfPolys() == 0)
	{
		std::cout << "Convert: No polygons can be created, probably all voxels are empty" << std::endl;
		//vtkDebugMacro("Convert: No polygons can be created, probably all voxels are empty");
		//closedSurfacePolyData->Reset();
		//return true;
		return vtkSmartPointer<vtkPolyData>::New();
	}

	// Perform smoothing using specified factor
	double smoothingFactor = 0.15;
	vtkSmartPointer<vtkSmoothPolyDataFilter> smoothFilter = vtkSmartPointer<vtkSmoothPolyDataFilter>::New();
	smoothFilter->SetInputData(processingResult);
	smoothFilter->SetRelaxationFactor(smoothingFactor);
	smoothFilter->Update();
	processingResult = smoothFilter->GetOutput();

	return processingResult;
}

vtkSmartPointer<vtkPolyData> MeshExtraction::CreateMesh(ImageType::Pointer image, double isoValue, bool isSmoothing)
{
	//image = ImageIO::Threshold(image, isoValue, 10000);

	vtkSmartPointer<vtkImageData> imageData;
	imageData = ConvertItkToVtkImage(image);

	std::cout << "iso value: " << isoValue << std::endl;

	double fractionalOversamplingFactor = 1;

	// Resize the image with interpolation, this helps the conversion for structures with small labelmaps
	vtkSmartPointer<vtkImageResize> imageResize = vtkSmartPointer<vtkImageResize>::New();
	imageResize->SetInputData(imageData);
	imageResize->BorderOn();
	imageResize->SetResizeMethodToMagnificationFactors();
	imageResize->SetMagnificationFactors(fractionalOversamplingFactor, fractionalOversamplingFactor, fractionalOversamplingFactor);
	imageResize->InterpolateOn();
	imageResize->Update();
	imageData = imageResize->GetOutput();

	vtkSmartPointer<vtkFlyingEdges3D> meshgenerator = vtkSmartPointer<vtkFlyingEdges3D>::New();
	//vtkSmartPointer<vtkMarchingCubes> meshgenerator = vtkSmartPointer<vtkMarchingCubes>::New();
	meshgenerator->SetInputData(imageData);
	meshgenerator->SetNumberOfContours(1);
	meshgenerator->SetValue(0, isoValue);
	meshgenerator->ComputeScalarsOff();
	meshgenerator->ComputeGradientsOff();
	meshgenerator->ComputeNormalsOff();
	meshgenerator->Update();

	vtkSmartPointer<vtkPolyData> processingResult = meshgenerator->GetOutput();

	if (processingResult->GetNumberOfPolys() == 0)
	{
		std::cout << "Convert: No polygons can be created, probably all voxels are empty" << std::endl;
		//vtkDebugMacro("Convert: No polygons can be created, probably all voxels are empty");
		//closedSurfacePolyData->Reset();
		//return true;
		return vtkSmartPointer<vtkPolyData>::New();
	}

	if (isSmoothing)
	{
		// Smoothing 1
		// Perform smoothing using specified factor
		double lpsmoothingFactor = 0.2;
		std::cout << "vtkSmoothPolyDataFilter smoothing factor: " << lpsmoothingFactor << std::endl;
		vtkSmartPointer<vtkSmoothPolyDataFilter> smoothFilter = vtkSmartPointer<vtkSmoothPolyDataFilter>::New();
		smoothFilter->SetInputData(processingResult);
		smoothFilter->SetRelaxationFactor(lpsmoothingFactor);
		smoothFilter->FeatureEdgeSmoothingOn();
		smoothFilter->Update();
		processingResult = smoothFilter->GetOutput();
	}

	// Another smoother version
	/*double smoothingFactor = 0.5;
	std::cout << "vtkWindowedSincPolyDataFilter smoothing factor: " << smoothingFactor << std::endl;
	vtkSmartPointer<vtkWindowedSincPolyDataFilter> smoother = vtkSmartPointer<vtkWindowedSincPolyDataFilter>::New();
	smoother->SetInputData(processingResult);
	//smoother->SetNumberOfIterations(20); // based on VTK documentation ("Ten or twenty iterations is all the is usually necessary")
										 // This formula maps:
										 // 0.0  -> 1.0   (almost no smoothing)
										 // 0.25 -> 0.1   (average smoothing)
										 // 0.5  -> 0.01  (more smoothing)
										 // 1.0  -> 0.001 (very strong smoothing)
	double passBand = pow(10.0, -4.0*smoothingFactor);
	smoother->SetPassBand(passBand);
	smoother->BoundarySmoothingOff();
	smoother->FeatureEdgeSmoothingOn();
	smoother->NonManifoldSmoothingOn();
	smoother->NormalizeCoordinatesOn();
	smoother->Update();
	processingResult = smoother->GetOutput();*/

	return processingResult;
}

void MeshExtraction::WriteSTL(vtkSmartPointer<vtkPolyData> polyData, std::string path)
{
	std::cout << "Stored: " << path << std::endl;

	vtkSmartPointer<vtkSTLWriter> writer = vtkSmartPointer<vtkSTLWriter>::New();
	writer->SetFileName(path.c_str());
	writer->SetInputData(polyData);
	writer->Write();
}


vtkSmartPointer<vtkPolyData> MeshExtraction::ReadSTL(std::string path)
{
	std::cout << "Read: " << path << std::endl;

	vtkSmartPointer<vtkSTLReader> reader = vtkSmartPointer<vtkSTLReader>::New();
	reader->SetFileName(path.c_str());
	reader->Update();

	return reader->GetOutput();
}

void MeshExtraction::WriteMeshFile(MeshType::Pointer mesh, std::string fileName)
{
	try
	{
		std::cout << "Saving mesh to " << fileName << std::endl;

		typedef itk::MeshFileWriter<MeshType> WriterType;

		WriterType::Pointer writer = WriterType::New();

		writer->SetFileName(fileName);
		writer->SetInput(mesh);

		writer->Update();
	}
	catch (itk::ExceptionObject & err)
	{
		std::cerr << "ExceptionObject caught !" << std::endl;
		std::cerr << err << std::endl;
	}
}