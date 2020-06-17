/*=========================================================================
*
*  Copyright Insight Software Consortium
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*         http://www.apache.org/licenses/LICENSE-2.0.txt
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
*=========================================================================*/
#ifndef __itkGPUBinaryMorphologyImageFilter_h
#define __itkGPUBinaryMorphologyImageFilter_h

#include "itkGPUImageToImageFilter.h"
#include "itkZeroFluxNeumannBoundaryCondition.h"

namespace itk
{
	template< typename TInputImage, typename TOutputImage, typename TKernel >
	class GPUBinaryMorphologyImageFilter :
		public GPUImageToImageFilter< TInputImage, TOutputImage >
	{
		/** Standard "Self" & Superclass typedef. */
		typedef GPUBinaryMorphologyImageFilter                                     Self;
		typedef GPUImageToImageFilter< TInputImage, TOutputImage > GPUSuperclass;
		typedef SmartPointer< Self >                                                   Pointer;
		typedef SmartPointer< const Self >                                             ConstPointer;

		/** Extract some information from the image types.  Dimensionality
		* of the two images is assumed to be the same. */
		typedef typename TOutputImage::PixelType         OutputPixelType;
		typedef typename TOutputImage::InternalPixelType OutputInternalPixelType;
		typedef typename  TInputImage::PixelType         InputPixelType;
		typedef typename  TInputImage::InternalPixelType InputInternalPixelType;
		typedef typename TKernel::PixelType              OperatorValueType;

		typedef typename NumericTraits<InputPixelType>::ValueType InputPixelValueType;
		typedef typename NumericTraits<OutputPixelType>::RealType ComputingPixelType;


		/** ImageDimension constants */
		itkStaticConstMacro(ImageDimension, unsigned int, TInputImage::ImageDimension);
		itkStaticConstMacro(InputImageDimension, unsigned int, TInputImage::ImageDimension);
		itkStaticConstMacro(OutputImageDimension, unsigned int, TOutputImage::ImageDimension);

		typedef GPUImage<OperatorValueType, itkGetStaticConstMacro(ImageDimension)> NeighborhoodGPUBufferType;

		/** Image typedef support. */
		typedef TInputImage                      InputImageType;
		typedef TOutputImage                     OutputImageType;
		typedef typename InputImageType::Pointer InputImagePointer;

		/** Typedef for generic boundary condition pointer. */
		typedef ImageBoundaryCondition< InputImageType > *
			ImageBoundaryConditionPointerType;

		/** Typedef for the default boundary condition */
		typedef ZeroFluxNeumannBoundaryCondition< InputImageType > DefaultBoundaryCondition;

		/** Superclass typedefs. */
		typedef typename GPUSuperclass::OutputImageRegionType OutputImageRegionType;

	public:
		GPUBinaryMorphologyImageFilter();

		void SetKernel(const TKernel&);

		itkSetMacro(ForegroundValue, InputPixelType);
		itkGetConstMacro(ForegroundValue, InputPixelType);

		itkSetMacro(BackgroundValue, InputPixelType);
		itkGetConstMacro(BackgroundValue, InputPixelType);

		itkSetMacro(BoundaryToForeground, bool);
		itkGetConstMacro(BoundaryToForeground, bool);

		const TKernel& GetKernel() const;

	protected:
		typename NeighborhoodGPUBufferType::Pointer m_NeighborhoodGPUBuffer;
		TKernel m_kernel;
		int m_seCount;
		InputPixelType m_ForegroundValue;
		InputPixelType m_BackgroundValue;
		bool m_BoundaryToForeground;
	};

} /* itk namespace */

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkGPUBinaryMorphologyImageFilter.hxx"
#endif

#endif
