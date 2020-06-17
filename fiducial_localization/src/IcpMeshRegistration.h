#ifndef __IcpMeshRegistration_h
#define __IcpMeshRegistration_h

#include "itkEuler3DTransform.h"
#include "itkVersorRigid3DTransform.h"
#include "itkTranslationTransform.h"
#include "itkEuclideanDistancePointMetric.h"
#include "itkLevenbergMarquardtOptimizer.h"
#include "itkPointSetToPointSetRegistrationMethod.h"
#include "itkDanielssonDistanceMapImageFilter.h"
#include "itkPointSetToImageFilter.h"
#include "itkCenteredTransformInitializer.h"
#include "itkTransformMeshFilter.h"

#include "ImageIO.h"
#include "MeshExtraction.h"

class IcpMeshRegistration
{

public:
	typedef MeshExtraction::MeshType MeshType;

	typedef ImageIO::ImageType ImageType;

	//typedef itk::VersorRigid3DTransform<double> TransformType;
	typedef itk::Euler3DTransform<double> TransformType;
	//typedef itk::TranslationTransform<double, 3> TransformType;

	typedef itk::TranslationTransform<double, 3> CentroidTranslationTransformType;

public:

	static TransformType::Pointer IcpWrap(MeshType::Pointer fixedPointSet, MeshType::Pointer movingPointSet, ImageType::Pointer image);

	static TransformType::Pointer Icp(MeshType::Pointer fixedPointSet, MeshType::Pointer movingPointSet, ImageType::Pointer image, TransformType::Pointer preRotationTransform);

	static MeshType::Pointer TransformMesh(MeshType::Pointer mesh, TransformType::Pointer transform);


	static double RMSE;
	static double NumberOfIterations;
	static double Time;

protected:
	IcpMeshRegistration() {}
	~IcpMeshRegistration() {}
};

class CommandIterationUpdate : public itk::Command
{
public:
	typedef  CommandIterationUpdate   Self;
	typedef  itk::Command             Superclass;
	typedef itk::SmartPointer<Self>   Pointer;
	itkNewMacro(Self);
protected:
	CommandIterationUpdate() {};
public:
	typedef itk::LevenbergMarquardtOptimizer     OptimizerType;
	typedef const OptimizerType *                OptimizerPointer;
	void Execute(itk::Object *caller, const itk::EventObject & event)
	{
		Execute((const itk::Object *)caller, event);
	}

	int counter = 0;

	void Execute(const itk::Object * object, const itk::EventObject & event)
	{
		OptimizerPointer optimizer =
			dynamic_cast<OptimizerPointer>(object);
		if (!itk::IterationEvent().CheckEvent(&event))
		{
			return;
		}

		++counter;
		//std::cout << "Value = " << optimizer->GetCachedValue() << std::endl;
		//std::cout << "Iteration = " << ++counter << " Position = " << optimizer->GetCachedCurrentPosition() << std::endl;
		//std::cout << "Iteration = " << ++counter << std::endl;
		//std::cout << std::endl;
	}
};




#endif