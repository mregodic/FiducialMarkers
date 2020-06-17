
#include "VtkMeshSurfaceExtraction.h"

#include <vtkPolyDataConnectivityFilter.h>
#include <vtkImageResize.h>
#include <vtkWindowedSincPolyDataFilter.h>

VtkMeshSurfaceExtraction::MeshType::Pointer VtkMeshSurfaceExtraction::ReadMeshFile(std::string fileName)
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

vtkSmartPointer<vtkImageData> VtkMeshSurfaceExtraction::ConvertItkToVtkImage(ImageType::Pointer image)
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

vtkSmartPointer<vtkImageData> VtkMeshSurfaceExtraction::ConvertItkToVtkImageBinary(BinaryImageType::Pointer image)
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

vtkSmartPointer<vtkPolyData> VtkMeshSurfaceExtraction::CreateMeshBinary(BinaryImageType::Pointer image)
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

vtkSmartPointer<vtkPolyData> VtkMeshSurfaceExtraction::CreateMesh(ImageType::Pointer image, double isoValue, int idx, bool isSmoothing)
{
	image = ImageIO::Threshold(image, 1500, 50945);

	isoValue = 1500;

	vtkSmartPointer<vtkImageData> vtkImageData;
	vtkImageData = ConvertItkToVtkImage(image);

	std::cout << "iso value: " << isoValue << std::endl;

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


	vtkSmartPointer<vtkPolyDataConnectivityFilter> connectivityFilter =
		vtkSmartPointer<vtkPolyDataConnectivityFilter>::New();
	connectivityFilter->SetInputData(processingResult);
	
	//double* p = new double[3]{ 1.21752, 0.343643, 47.8174 };
	//connectivityFilter->SetClosestPoint(p);
	//connectivityFilter->SetExtractionModeToClosestPointRegion();

	connectivityFilter->SetExtractionModeToLargestRegion();

	//if (idx == 12)
	//{
	//	connectivityFilter->AddSpecifiedRegion(0);
	//	connectivityFilter->SetExtractionModeToSpecifiedRegions();
	//}
	//else
	//{
	//	connectivityFilter->SetExtractionModeToLargestRegion();
	//}

	connectivityFilter->Update();

	// this can be done by running the ICP on this mesh and then setting the closest point which is [0, 0, 0] in the reference mesh
	//connectivityFilter->SetClosestPoint()
	//connectivityFilter->SetExtractionModeToClosestPointRegion();

	processingResult = connectivityFilter->GetOutput();

	if (isSmoothing)
	{
		// Smoothing 1
		// Perform smoothing using specified factor
		//double smoothingFactor = 0.05; // all screws with spacing 0.5
		double lpsmoothingFactor = 0.2;
		//double smoothingFactor = 0.1; // 3 mm
		//double smoothingFactor = 0.22; // 3.7 mm
		//double smoothingFactor = 0.26; // 4.5 mm
		std::cout << "vtkSmoothPolyDataFilter smoothing factor: " << lpsmoothingFactor << std::endl;
		vtkSmartPointer<vtkSmoothPolyDataFilter> smoothFilter = vtkSmartPointer<vtkSmoothPolyDataFilter>::New();
		smoothFilter->SetInputData(processingResult);
		smoothFilter->SetRelaxationFactor(lpsmoothingFactor);
		smoothFilter->FeatureEdgeSmoothingOn();
		smoothFilter->Update();
		//std::cout << "is on: " << smoothFilter->GetFeatureEdgeSmoothing() << std::endl;
		processingResult = smoothFilter->GetOutput();
	}


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

void VtkMeshSurfaceExtraction::WriteSTL(vtkSmartPointer<vtkPolyData> polyData, std::string path)
{
	std::cout << "Stored: " << path << std::endl;

	vtkSmartPointer<vtkSTLWriter> writer = vtkSmartPointer<vtkSTLWriter>::New();
	writer->SetFileName(path.c_str());
	writer->SetInputData(polyData);
	writer->Write();
}


vtkSmartPointer<vtkPolyData> VtkMeshSurfaceExtraction::ReadSTL(std::string path)
{
	std::cout << "Read: " << path << std::endl;

	vtkSmartPointer<vtkSTLReader> reader = vtkSmartPointer<vtkSTLReader>::New();
	reader->SetFileName(path.c_str());
	reader->Update();

	return reader->GetOutput();
}

void VtkMeshSurfaceExtraction::TestMesh(ImageType::Pointer image)
{
	image = ImageIO::Threshold(image, 1500, 5945);

	//BinaryImageType::Pointer binaryImage = ImageIO::BinarizeImage(image, 1, 5945);

	std::cout << "Image size " << image->GetLargestPossibleRegion().GetSize() << std::endl;

	vtkSmartPointer<vtkImageData> vtkImageData;
	vtkImageData = ConvertItkToVtkImage(image);
	//vtkImageData = ConvertItkToVtkImageBinary(binaryImage);

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

	double isoValue = 1;

	double fractionalThreshold = 0;
	double minimumValue = 1486;
	double maximumValue = 5945;

	//isoValue = (fractionalThreshold * (maximumValue - minimumValue)) + minimumValue;

	std::cout << "iso value: " << isoValue << std::endl;

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
		return;
	}

	//double decimationFactor = 0;
	// Decimate if necessary
	//vtkSmartPointer<vtkDecimatePro> decimator = vtkSmartPointer<vtkDecimatePro>::New();
	//decimator->SetInputData(processingResult);
	//if (decimationFactor > 0.0)
	//{
	//	decimator->SetFeatureAngle(60);
	//	decimator->SplittingOff();
	//	decimator->PreserveTopologyOn();
	//	decimator->SetMaximumError(1);
	//	decimator->SetTargetReduction(decimationFactor);
	//	try
	//	{
	//		decimator->Update();
	//		processingResult = decimator->GetOutput();
	//	}
	//	catch (...)
	//	{
	//		//vtkErrorMacro("Error decimating model");
	//		//return false;
	//		std::cout << "cannot decimate" << std::endl;
	//		return;
	//	}
	//}

	// Perform smoothing using specified factor
	double smoothingFactor = 0.05;
	vtkSmartPointer<vtkSmoothPolyDataFilter> smoothFilter = vtkSmartPointer<vtkSmoothPolyDataFilter>::New();
	smoothFilter->SetInputData(processingResult);
	smoothFilter->SetRelaxationFactor(smoothingFactor);
	smoothFilter->Update();
	processingResult = smoothFilter->GetOutput();

	// Write the file
	vtkSmartPointer<vtkSTLWriter> writer = vtkSmartPointer<vtkSTLWriter>::New();
	writer->SetFileName("E:\\meshes\\vtk_mesh.stl");
	writer->SetInputData(processingResult);
	writer->Write();
}
