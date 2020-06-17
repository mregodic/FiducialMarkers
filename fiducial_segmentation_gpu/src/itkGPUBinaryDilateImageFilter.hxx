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
#ifndef __itkGPUBinaryDilateImageFilter_hxx
#define __itkGPUBinaryDilateImageFilter_hxx

#include "itkGPUBinaryDilateImageFilter.h"
#include "itkProgressReporter.h"
#include "itkOpenCLUtil.h"

namespace itk
{

	template< typename TInputImage, typename TOutputImage, typename TKernel >
	GPUBinaryDilateImageFilter< TInputImage, TOutputImage, TKernel >
		::GPUBinaryDilateImageFilter()
	{
		std::ostringstream defines;

		if (TInputImage::ImageDimension > 3 || TInputImage::ImageDimension < 1)
		{
			itkExceptionMacro("GPUBinaryDilateImageFilter supports 1/2/3D image.");
		}

		defines << "#define DIM_" << TInputImage::ImageDimension << "\n";

		defines << "#define INTYPE ";
		GetTypenameInString(typeid (typename TInputImage::PixelType), defines);

		defines << "#define OUTTYPE ";
		GetTypenameInString(typeid (typename TOutputImage::PixelType), defines);

		defines << "#define OPTYPE ";
		GetTypenameInString(typeid (typename TKernel::PixelType), defines);

		defines << "#define BOOL ";
		GetTypenameInString(typeid(unsigned char), defines);

		std::cout << "Defines: " << defines.str() << std::endl;

		const char* GPUSource = GPUBinaryDilateImageFilter::GetOpenCLSource();

		// load and build program
		this->m_GPUKernelManager->LoadProgramFromString( GPUSource, defines.str().c_str() );

		// create kernel
		m_BinaryDilateFilterGPUKernelHandle = this->m_GPUKernelManager->CreateKernel("BinaryDilateFilter");
	}


	template< typename TInputImage, typename TOutputImage, typename TKernel >
	void
		GPUBinaryDilateImageFilter< TInputImage, TOutputImage, TKernel >
		::GPUGenerateData()
	{
		ProgressReporter progress(this, 0, 1, 1);

		int kHd = m_BinaryDilateFilterGPUKernelHandle;

		typedef typename GPUTraits< TInputImage >::Type  GPUInputImage;
		typedef typename GPUTraits< TOutputImage >::Type GPUOutputImage;
		typedef GPUImageDataManager<GPUInputImage>            GPUInputManagerType;
		typedef GPUImageDataManager<GPUOutputImage>           GPUOutputManagerType;

		typename GPUInputImage::Pointer  inPtr = dynamic_cast<GPUInputImage *>(this->ProcessObject::GetInput(0));
		typename GPUOutputImage::Pointer otPtr = dynamic_cast<GPUOutputImage *>(this->ProcessObject::GetOutput(0));

		typename GPUOutputImage::SizeType outSize = otPtr->GetBufferedRegion().GetSize();

		int radius[3];
		int imgSize[3];

		radius[0] = radius[1] = radius[2] = 0;
		imgSize[0] = imgSize[1] = imgSize[2] = 1;

		int ImageDim = (int)TInputImage::ImageDimension;

		for (int i = 0; i < ImageDim; i++)
		{
			radius[i] = (this->GetKernel()).GetRadius(i);
			imgSize[i] = outSize[i];
		}

		size_t localSize[3], globalSize[3];
		localSize[0] = localSize[1] = localSize[2] = OpenCLGetLocalBlockSize(ImageDim);
		for (int i = 0; i < ImageDim; i++)
		{
			globalSize[i] = localSize[i] * (unsigned int)ceil((float)outSize[i] / (float)localSize[i]); //
														 // total
														 // #
														 // of
														 // threads
		}

		// arguments set up
		cl_uint argidx = 0;
		this->m_GPUKernelManager->template SetKernelArgWithImageAndBufferedRegion<GPUInputManagerType>
			(kHd, argidx, inPtr->GetDataManager());
		this->m_GPUKernelManager->template SetKernelArgWithImageAndBufferedRegion<GPUOutputManagerType>
			(kHd, argidx, otPtr->GetDataManager());
		this->m_GPUKernelManager->SetKernelArgWithImage(kHd, argidx++, this->m_NeighborhoodGPUBuffer->GetGPUDataManager());

		for (int i = 0; i < (int)TInputImage::ImageDimension; i++)
		{
			this->m_GPUKernelManager->SetKernelArg(kHd, argidx++, sizeof(int), &(radius[i]));
		}


		this->m_GPUKernelManager->SetKernelArg(kHd, argidx++, sizeof(InputPixelType), &(this->m_ForegroundValue));
		this->m_GPUKernelManager->SetKernelArg(kHd, argidx++, sizeof(InputPixelType), &(this->m_BackgroundValue));

		unsigned char borderFg = this->m_BoundaryToForeground ? 1 : 0;
		this->m_GPUKernelManager->SetKernelArg(kHd, argidx++, sizeof(unsigned char), &(borderFg));

		// launch kernel
		this->m_GPUKernelManager->LaunchKernel(kHd, ImageDim, globalSize, localSize);

		progress.CompletedPixel();
	}

} // end namespace itk

#endif
