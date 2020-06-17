
#include <iostream>
#include <string>

#include "ImageIO.h"
#include "MeshExtraction.h"
#include "IcpMeshRegistration.h"

typedef ImageIO::ImageType ImageType;
typedef ImageIO::BinaryImageType BinaryImageType;
typedef MeshExtraction::MeshType MeshType;
typedef MeshExtraction::VtkMeshType VtkMeshPointer;
typedef IcpMeshRegistration::TransformType TransformType;
typedef TransformType::InputPointType PointType;

int main()
{
	// spherical fiducial example
	const std::string inputFile = "F:\\test_data\\spherical_fiducials\\A3_D2_L2.mhd";
	const std::string referenceMeshFile = "F:\\test_data\\reference_mesh\\rhinospider.stl";
	const std::string outputMeshFile = "F:\\test_data\\mesh_outputs\\A3_D2_L2.vtk";

	// screw example
	//const std::string inputFile = "F:\\test_data\\screws\\CONRAD_SCREW_4p5mm_L14.mhd";
	//const std::string referenceMeshFile = "F:\\test_data\\reference_mesh\\single_screw_3x4p5.stl";
	//const std::string outputMeshFile = "F:\\test_data\\mesh_outputs\\CONRAD_SCREW_4p5mm_L14.vtk";
	
	const std::string tmp_mesh_path = "F:\\test_data\\tmp_mesh\\vtkmesh.stl";

	double isoValue = 120;
	double applySmoothing = true;
	TransformType::InputPointType refFiducialPosition = PointType(new double[3]{ 0, 0, 0 });

	ImageType::Pointer image = ImageIO::ReadImageFile(inputFile);
	MeshType::Pointer referenceMesh = MeshExtraction::ReadMeshFile(referenceMeshFile);
	std::cout << "read file" << std::endl;

	VtkMeshPointer vtkMesh = MeshExtraction::CreateMesh(image, isoValue, applySmoothing);

	if (vtkMesh->GetNumberOfPolys() > 0)
	{
		// Write to vtk mesh and then read as an itk mesh
		MeshExtraction::WriteSTL(vtkMesh, tmp_mesh_path);
		MeshType::Pointer imageMesh = MeshExtraction::ReadMeshFile(tmp_mesh_path);

		// Output transform
		TransformType::Pointer transform = TransformType::New();
		transform = IcpMeshRegistration::IcpWrap(imageMesh, referenceMesh, image);

		// Obtain fiducial position
		PointType detectedPosition = transform->TransformPoint(refFiducialPosition);
		std::cout << "Detected position: " << detectedPosition << std::endl;

		// Store the transformed mesh
		TransformType::Pointer inverseTransform = TransformType::New();
		inverseTransform->SetIdentity();
		transform->GetInverse(inverseTransform);

		MeshExtraction::WriteMeshFile(IcpMeshRegistration::TransformMesh(imageMesh, inverseTransform), outputMeshFile);
	}
	else
	{
		std::cout << "EMPTY MESH!!!" << std::endl;
	}

	int a;
	std::cin >> a;
}

