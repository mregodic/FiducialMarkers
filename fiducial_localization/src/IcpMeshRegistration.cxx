
#include "IcpMeshRegistration.h"
#include <chrono> 
#include <thread>

double IcpMeshRegistration::RMSE = 0;
double IcpMeshRegistration::NumberOfIterations = 0;
double IcpMeshRegistration::Time = 0;

IcpMeshRegistration::TransformType::Pointer IcpMeshRegistration::IcpWrap(MeshType::Pointer fixedPointSet, MeshType::Pointer movingPointSet, ImageType::Pointer image)
{
	double y = 0;

	double end = 180; // 360
	double increase = 30; // 10

	double bestRMS = itk::NumericTraits<double>::max();
	double bestAngle = 0;

	// Run multiple ICPs with different pitch angle.
	while (y <= end)
	{
		TransformType::Pointer rotateTransform = TransformType::New();
		rotateTransform->SetIdentity();
		rotateTransform->SetRotation(0, y, 0);

		Icp(fixedPointSet, movingPointSet, image, rotateTransform);

		if (IcpMeshRegistration::RMSE < bestRMS)
		{
			bestRMS = IcpMeshRegistration::RMSE;
			bestAngle = y;
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(1));

		y += increase;
	}

	std::cout << "Best RMSE = " << bestRMS << " - best angle = " << bestAngle << std::endl;

	TransformType::Pointer rotateTransform = TransformType::New();
	rotateTransform->SetIdentity();
	rotateTransform->SetRotation(0, bestAngle, 0);

	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	TransformType::Pointer  transform = Icp(fixedPointSet, movingPointSet, image, rotateTransform);
	
	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
	Time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

	//std::cout << "Number of iterations is " << IcpMeshRegistration::NumberOfIterations << ", and duration is " << duration << " ms" << std::endl;

	return transform;
}

IcpMeshRegistration::TransformType::Pointer IcpMeshRegistration::Icp(MeshType::Pointer fixedPointSet, MeshType::Pointer movingPointSet, ImageType::Pointer image, TransformType::Pointer preRotationTransform)
{
	// Define points

	using PointSetType = MeshType;
	//using PointSetType = itk::PointSet< MeshType::PointType::ValueType, MeshType::PointType::Dimension >;
	using PointType = PointSetType::PointType;

	// Define metric
	using MetricType = itk::EuclideanDistancePointMetric<PointSetType, PointSetType>;
	MetricType::Pointer metric = MetricType::New();

	// Add spacing
	using PointsToImageFilterType = itk::PointSetToImageFilter<PointSetType, ImageType>;
	PointsToImageFilterType::Pointer pointsToImageFilter = PointsToImageFilterType::New();
	pointsToImageFilter->SetInput(fixedPointSet);

	ImageType::SpacingType spacing;
	spacing = image->GetSpacing();
	//std::cout << spacing << std::endl;

	ImageType::PointType origin;
	origin = image->GetOrigin();
	pointsToImageFilter->SetDirection(image->GetDirection());

	pointsToImageFilter->SetSpacing(spacing);
	pointsToImageFilter->SetOrigin(origin);

	pointsToImageFilter->Update();
	ImageType::Pointer imagePoints = pointsToImageFilter->GetOutput();

	// This sometimes takes very long time to complete for ICP
	/*using DistanceImageType = itk::Image<unsigned short, PointType::Dimension>;
	using DistanceFilterType = itk::DanielssonDistanceMapImageFilter<ImageType, DistanceImageType>;
	DistanceFilterType::Pointer distanceFilter = DistanceFilterType::New();
	distanceFilter->SetInput(imagePoints);
	distanceFilter->Update();
	metric->SetDistanceMap(distanceFilter->GetOutput());*/

	// Define optimizer
	using OptimizerType = itk::LevenbergMarquardtOptimizer;
	OptimizerType::Pointer optimizer = OptimizerType::New();

	//optimizer->SetUseCostFunctionGradient(false);
	optimizer->UseCostFunctionGradientOff(); // produces bad results

	CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
	optimizer->AddObserver(itk::IterationEvent(), observer);

	using RegistrationType = itk::PointSetToPointSetRegistrationMethod<PointSetType, PointSetType>;

	RegistrationType::Pointer registration = RegistrationType::New();

	TransformType::Pointer transform = TransformType::New();
	transform->SetIdentity();

	OptimizerType::ScalesType scales(transform->GetNumberOfParameters());
	constexpr double translationScale = 1000.0; // dynamic range of translations
	constexpr double rotationScale = 1.0;       // dynamic range of rotations
	scales[0] = 1.0 / rotationScale;
	scales[1] = 1.0 / rotationScale;
	scales[2] = 1.0 / rotationScale;
	scales[3] = 1.0 / translationScale;
	scales[4] = 1.0 / translationScale;
	scales[5] = 1.0 / translationScale;

	// pre transform
	CentroidTranslationTransformType::Pointer preTransform = CentroidTranslationTransformType::New();
	preTransform->SetIdentity();
	//preTransform->SetParameters(preRotationTransform->GetParameters());
	OptimizerType::ScalesType prescales(preTransform->GetNumberOfParameters());
	prescales[0] = 1.0 / translationScale;
	prescales[1] = 1.0 / translationScale;
	prescales[2] = 1.0 / translationScale;

	constexpr unsigned long numberOfIterations = 1000; //10000
	// these are very good set on experimental data
	const double gradientTolerance = 1e-5;    // convergence criterion  //1e-5
	const double valueTolerance = 1e-5;    // convergence criterion // 1e-5
	const double epsilonFunction = 1e-16;   // convergence criterion // 1e-6 //16

	optimizer->SetScales(prescales);
	optimizer->SetNumberOfIterations(numberOfIterations);
	optimizer->SetValueTolerance(valueTolerance);
	optimizer->SetGradientTolerance(gradientTolerance);
	optimizer->SetEpsilonFunction(epsilonFunction);
	//std::cout << "test 1" << std::endl;

	// PRE REGISTRATION to FIND PRE TRANSFORMATION
	registration->SetInitialTransformParameters(preTransform->GetParameters());
	registration->SetMetric(metric);
	registration->SetOptimizer(optimizer);
	registration->SetTransform(preTransform);
	registration->SetFixedPointSet(fixedPointSet);
	registration->SetMovingPointSet(movingPointSet);

	try
	{
		registration->Update();
	}
	catch (itk::ExceptionObject & e)
	{
		std::cout << e << std::endl;
	}

	// FINAL REGISTRATION
	//std::cout << "starting second registration" << std::endl;

	optimizer->SetScales(scales);

	// Applying pre transformation
	TransformType::ParametersType initialParameters = transform->GetParameters();
	CentroidTranslationTransformType::ParametersType preParameters = preTransform->GetParameters();
	initialParameters[3] = preParameters[0];
	initialParameters[4] = preParameters[1];
	initialParameters[5] = preParameters[2];
	transform->SetParameters(initialParameters);
	transform->SetRotation(preRotationTransform->GetAngleX(), preRotationTransform->GetAngleY(), preRotationTransform->GetAngleZ());

	registration->SetInitialTransformParameters(transform->GetParameters());
	registration->SetMetric(metric);
	registration->SetOptimizer(optimizer);
	registration->SetTransform(transform);
	registration->SetFixedPointSet(fixedPointSet);
	registration->SetMovingPointSet(movingPointSet);

	try
	{
		metric->ComputeSquaredDistanceOn();

		registration->ResetPipeline();
		registration->Update();
	}
	catch (itk::ExceptionObject & e)
	{
		std::cout << e << std::endl;
	}


	//std::cout << std::endl;
	//std::cout << metric->GetValue(transform->GetParameters()) << std::endl;

	//std::cout << "Solution = " << transform->GetParameters() << std::endl;
	//std::cout << "Solution = " << metric->GetComputeSquaredDistance() << std::endl;

	PointSetType::Pointer sortedSources;

	using CellDataIterator = MeshType::PointsContainerIterator;

	using PointsContainer = MeshType::PointsContainer;

	PointsContainer* fixedPoints = movingPointSet->GetPoints();
	PointsContainer* movingPoints = fixedPointSet->GetPoints();

	double m_ErrorMean = 0;

	int counter = 0;


	//std::cout << "Size fixed" << fixedPoints->size() << std::endl;
	//std::cout << "Size moving" << movingPoints->size() << std::endl;

	PointType meanPosition;
	meanPosition[0] = 0;
	meanPosition[1] = 0;
	meanPosition[2] = 0;
	for (PointsContainer::iterator fixedIt = fixedPoints->begin(); fixedIt != fixedPoints->end(); ++fixedIt)
	{
		meanPosition[0] += (*fixedIt)[0];
		meanPosition[1] += (*fixedIt)[1];
		meanPosition[2] += (*fixedIt)[2];
	}
	meanPosition[0] /= fixedPoints->size();
	meanPosition[1] /= fixedPoints->size();
	meanPosition[2] /= fixedPoints->size();
	//std::cout << "fixed mean position: " << meanPosition << std::endl;

	meanPosition[0] = 0;
	meanPosition[1] = 0;
	meanPosition[2] = 0;
	for (PointsContainer::const_iterator movingIt = movingPoints->begin(); movingIt != movingPoints->end(); ++movingIt)
	{
		meanPosition[0] += (*movingIt)[0];
		meanPosition[1] += (*movingIt)[1];
		meanPosition[2] += (*movingIt)[2];
	}
	meanPosition[0] /= movingPoints->size();
	meanPosition[1] /= movingPoints->size();
	meanPosition[2] /= movingPoints->size();
	//std::cout << "moving mean position: " << meanPosition << std::endl;

	for (PointsContainer::const_iterator movingIt = movingPoints->begin(); movingIt != movingPoints->end(); ++movingIt)
	{
		++counter;

		double minDistance = itk::NumericTraits<double>::max();
		PointsContainer::iterator minDistanceIterator = fixedPoints->end();

		//std::cout << *movingIt << std::endl;

		//continue;

		for (PointsContainer::iterator fixedIt = fixedPoints->begin(); fixedIt != fixedPoints->end(); ++fixedIt)
		{
			//std::cout << *fixedIt << std::endl;
			//continue;

			TransformType::OutputPointType transformedSource = transform->TransformPoint((*fixedIt));
			double dist = (*movingIt).EuclideanDistanceTo(transformedSource);
			//double dist = movingIt->EuclideanDistanceTo(*fixedIt);

			//std::cout << "target: " << *movingIt << ", source: " << *fixedIt << ", transformed source: " << transformedSource << ", dist: " << dist << std::endl;

			if (dist < minDistance)
			{
				minDistanceIterator = fixedIt;
				minDistance = dist;
			}

			if (minDistanceIterator == fixedPoints->end())
			{
				std::cout << "error" << std::endl;
				break;
			}
		}

		//std::cout << counter << " (dist: " << movingIt->EuclideanDistanceTo(transform->TransformPoint(*minDistanceIterator)) << ", minDist: " << minDistance << ")" << std::endl;
		m_ErrorMean += minDistance;
		//std::cout << std::endl;
	}

	double RMS = m_ErrorMean / movingPoints->size();

	std::cout << "RMS = " << RMS << std::endl;

	RMSE = RMS;

	NumberOfIterations = observer->counter;

	return transform;
}

IcpMeshRegistration::MeshType::Pointer IcpMeshRegistration::TransformMesh(MeshType::Pointer mesh, TransformType::Pointer transform)
{
	using TransformFilterType = itk::TransformMeshFilter<MeshType, MeshType, TransformType>;
	TransformFilterType::Pointer transformMesh = TransformFilterType::New();
	transformMesh->SetInput(mesh);
	transformMesh->SetTransform(transform);
	transformMesh->Update();

	return transformMesh->GetOutput();
}