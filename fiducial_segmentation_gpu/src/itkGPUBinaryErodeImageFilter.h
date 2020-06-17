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
#ifndef __itkGPUBinaryErodeImageFilter_h
#define __itkGPUBinaryErodeImageFilter_h

#include "itkGPUImage.h"
#include "itkGPUBinaryMorphologyImageFilter.h"

namespace itk
{
	/** \class GPUBinaryErodeImageFilter
	* \brief Applies a single BinaryErode to an image region
	* using the GPU.
	*
	* \ingroup GPUBinaryMorphology
	*/

	/** Create a helper GPU Kernel class for GPUBinaryErodeImageFilter */
	itkGPUKernelClassMacro(GPUBinaryErodeImageFilterKernel);

	template< typename TInputImage, typename TOutputImage, typename TKernel >
	class GPUBinaryErodeImageFilter :
		public GPUBinaryMorphologyImageFilter< TInputImage, TOutputImage, TKernel >
	{
	public:
		/** Standard "Self" & Superclass typedef. */
		typedef GPUBinaryErodeImageFilter                                     Self;
		typedef GPUBinaryMorphologyImageFilter< TInputImage, TOutputImage, TKernel >	SuperClass;
		typedef SmartPointer< Self >                                                   Pointer;
		typedef SmartPointer< const Self >                                             ConstPointer;
		typedef typename  TInputImage::PixelType         InputPixelType;

		/** Method for creation through the object factory. */
		itkNewMacro(Self);

		/** Run-time type information (and related methods). */
		itkTypeMacro(GPUBinaryErodeImageFilter, GPUBinaryMorphologyImageFilter);

		/** Get OpenCL Kernel source as a string, creates a GetOpenCLSource method */
		itkGetOpenCLSourceFromKernelMacro(GPUBinaryErodeImageFilterKernel);

	public:
#ifdef __GNUC__
		__attribute__((visibility("default")))
#endif
		void SetErodeValue(InputPixelType p)
		{
			this->m_ForegroundValue = p;
		}

		const InputPixelType& GetErodeValue()
		{
			return this->m_ForegroundValue;
		}

	protected:
		GPUBinaryErodeImageFilter();
		virtual ~GPUBinaryErodeImageFilter() {
		}

		void GPUGenerateData();

		void PrintSelf(std::ostream & os, Indent indent) const
		{
			SuperClass::PrintSelf(os, indent);
		}

	private:
		GPUBinaryErodeImageFilter(const Self &); //purposely not implemented
		void operator=(const Self &);                     //purposely not implemented

		int m_BinaryErodeFilterGPUKernelHandle;
	};
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkGPUBinaryErodeImageFilter.hxx"
#endif

#endif
