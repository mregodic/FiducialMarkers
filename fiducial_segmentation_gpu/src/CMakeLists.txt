project(SegmentFiducialMarkersGPU)

set(SegmentFiducialMarkersGPU_SRC
"SegmentFiducialMarkersGPU.cxx"
"itkGPUBinaryDilateImageFilter.h" 
"itkGPUBinaryDilateImageFilter.hxx" 
"itkGPUBinaryErodeImageFilter.h" 
"itkGPUBinaryErodeImageFilter.hxx" 
"itkGPUBinaryMorphologyImageFilter.h"
"itkGPUBinaryMorphologyImageFilter.hxx"
"GPUBinaryErodeImageFilter.cl"
"GPUBinaryDilateImageFilter.cl"
)

# Find ITK.
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(SegmentFiducialMarkersGPU  ${SegmentFiducialMarkersGPU_SRC})

target_link_libraries(SegmentFiducialMarkersGPU ${ITK_LIBRARIES})