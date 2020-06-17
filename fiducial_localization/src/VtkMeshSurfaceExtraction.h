#ifndef __VtkMeshSurfaceExtraction_h
#define __VtkMeshSurfaceExtraction_h

#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkPolyData.h>
#include <vtkSTLWriter.h>
#include <vtkSTLReader.h>
//#include <vtkMarchingCubes.h>
#include <vtkSmoothPolyDataFilter.h>
#include <vtkDecimatePro.h>
#include <vtkFlyingEdges3D.h>

#include <itkImageToVTKImageFilter.h>

#include "itkSTLMeshIOFactory.h"
#include "itkSTLMeshIO.h"
#include "itkMeshFileReader.h"


#include "ImageIO.h"

class VtkMeshSurfaceExtraction
{
public:
	typedef ImageIO::ImageType ImageType;

	typedef ImageIO::BinaryImageType BinaryImageType;

	typedef itk::Mesh<float, 3 > MeshType;

	typedef vtkSmartPointer<vtkPolyData> VtkMeshType;

public:
	static MeshType::Pointer ReadMeshFile(std::string fileName);

	static vtkSmartPointer<vtkImageData> ConvertItkToVtkImage(ImageType::Pointer image);

	static vtkSmartPointer<vtkImageData> ConvertItkToVtkImageBinary(BinaryImageType::Pointer image);

	static vtkSmartPointer<vtkPolyData> CreateMeshBinary(BinaryImageType::Pointer image);

	static vtkSmartPointer<vtkPolyData> CreateMesh(ImageType::Pointer image, double isoValue, int idx, bool isSmoothing);

	static void WriteSTL(vtkSmartPointer<vtkPolyData> polyData, std::string path);

	static vtkSmartPointer<vtkPolyData> VtkMeshSurfaceExtraction::ReadSTL(std::string path);

	static void TestMesh(ImageType::Pointer image);


protected:
	VtkMeshSurfaceExtraction() {}
	~VtkMeshSurfaceExtraction() {}

};



#endif