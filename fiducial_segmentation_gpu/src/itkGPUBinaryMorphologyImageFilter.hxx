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
#ifndef __itkGPUBinaryMorphologyImageFilter_hxx
#define __itkGPUBinaryMorphologyImageFilter_hxx

#include "itkGPUBinaryMorphologyImageFilter.h"

#include "itkImageRegionIterator.h"
#include "itkNumericTraits.h"
#include "itkOpenCLUtil.h"

namespace itk
{
template< typename TInputImage, typename TOutputImage, typename TKernel >
GPUBinaryMorphologyImageFilter< TInputImage, TOutputImage, TKernel >
::GPUBinaryMorphologyImageFilter()
{
    // Create GPU buffer to store neighborhood coefficient.
    // This will be used as __constant memory in the GPU kernel.
    m_NeighborhoodGPUBuffer = NeighborhoodGPUBufferType::New();

    m_seCount = 0;
    m_ForegroundValue = NumericTraits< InputPixelType >::max();
    m_BackgroundValue = NumericTraits< InputPixelType >::ZeroValue();
    m_BoundaryToForeground = false;
}

template< typename TInputImage, typename TOutputImage, typename TKernel >
void
GPUBinaryMorphologyImageFilter< TInputImage, TOutputImage, TKernel >
::SetKernel(const TKernel& p)
{
  m_kernel = p;

  /** Create GPU memory for operator coefficients */
  m_NeighborhoodGPUBuffer->Initialize();

  typename NeighborhoodGPUBufferType::IndexType  index;
  typename NeighborhoodGPUBufferType::SizeType   size;
  typename NeighborhoodGPUBufferType::RegionType region;

  for(int i=0; i<ImageDimension; i++)
    {
    index[i] = 0;
    size[i]  = (unsigned int)(p.GetSize(i) );
    }
  region.SetSize( size );
  region.SetIndex( index );
  region.Print(std::cout);

  m_NeighborhoodGPUBuffer->SetRegions( region );
  m_NeighborhoodGPUBuffer->Allocate();

  m_NeighborhoodGPUBuffer->GetLargestPossibleRegion().Print(std::cout);

  /** Copy coefficients */
  ImageRegionIterator<NeighborhoodGPUBufferType> iit(m_NeighborhoodGPUBuffer,
						     m_NeighborhoodGPUBuffer->GetLargestPossibleRegion() );

  std::cout << "iit.IsAtEnd(): " << iit.IsAtEnd() << std::endl;

  typename TKernel::ConstIterator nit = p.Begin();

  m_seCount = 0;
  for(iit.GoToBegin(); !iit.IsAtEnd(); ++iit, ++nit)
    {
    std::cout << (int) *nit << ",";
    iit.Set( static_cast< typename NeighborhoodGPUBufferType::PixelType >( *nit ) );
    if(*nit == 1)
	++m_seCount;
    }
  std::cout << "SE count: " << m_seCount << std::endl;

  /** Mark GPU dirty */
  m_NeighborhoodGPUBuffer->GetGPUDataManager()->SetGPUBufferDirty();
}

template< typename TInputImage, typename TOutputImage, typename TKernel >
const TKernel&
GPUBinaryMorphologyImageFilter< TInputImage, TOutputImage, TKernel >
::GetKernel() const
{
    return m_kernel;
}


} /* end of itk namespace */
#endif
