project(FiducialLocalization)

SET(FiducialLocalizationSRC
"FiducialLocalization.cxx"
"MeshExtraction.cxx"
"MeshExtraction.h"
"ImageIO.h"
"ImageIO.cxx"
"IcpMeshRegistration.h"
"IcpMeshRegistration.cxx"
)

# Find ITK.
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

# Find VTK.
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

add_executable(FiducialLocalization  ${FiducialLocalizationSRC})

target_link_libraries(FiducialLocalization ${ITK_LIBRARIES})
target_link_libraries(FiducialLocalization ${VTK_LIBRARIES})
