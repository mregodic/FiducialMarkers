#ifndef __MeshExtraction_h
#define __MeshExtraction_h

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
#include "itkMeshFileWriter.h"

#include "ImageIO.h"

class MeshExtraction
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

	static vtkSmartPointer<vtkPolyData> CreateMesh(ImageType::Pointer image, double isoValue, bool isSmoothing);

	static void WriteSTL(vtkSmartPointer<vtkPolyData> polyData, std::string path);

	static vtkSmartPointer<vtkPolyData> ReadSTL(std::string path);

	static void WriteMeshFile(MeshType::Pointer mesh, std::string fileName);

protected:
	MeshExtraction() {}
	~MeshExtraction() {}

};



#endif