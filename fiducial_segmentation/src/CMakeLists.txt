project(SegmentFiducialMarkers)

# Find ITK.
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})


add_executable(SegmentFiducialMarkers SegmentFiducialMarkers.cxx )

target_link_libraries(SegmentFiducialMarkers ${ITK_LIBRARIES})
