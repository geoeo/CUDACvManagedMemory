cmake_minimum_required(VERSION 3.25)

project(CUDACvManagedMemory LANGUAGES CXX)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(${CMAKE_PROJECT_NAME} SHARED)

target_include_directories(${CMAKE_PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    ${OpenCV_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_sources(${CMAKE_PROJECT_NAME}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_cv_managed_memory.cpp
  PUBLIC FILE_SET HEADERS
  BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/CUDACvManagedMemory/cuda_cv_managed_memory.hpp
)

target_link_libraries(${PROJECT_NAME}
	${OpenCV_LIBRARIES}
	${CUDA_LIBRARIES}
	)

### INSTALLING ###

# install the headers that we specified with FILE_SET (header files for the user to include in their projects)
install(TARGETS ${CMAKE_PROJECT_NAME}
  EXPORT CUDACvManagedMemoryTargets
  FILE_SET HEADERS
)


# provide some basic meta data about the library to generate the library's config file 
install(
  EXPORT CUDACvManagedMemoryTargets
  FILE CUDACvManagedMemoryTargets.cmake
  DESTINATION "lib/cmake/CUDACvManagedMemory"
)

# generate the library's config file
include(CMakePackageConfigHelpers)

set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR}/CUDACvManagedMemory
    CACHE PATH "Location of header files" )
    
set(SYSCONFIG_INSTALL_DIR ${CMAKE_INSTALL_BINDIR}
    CACHE PATH "Location of configuration files" )

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CUDACvManagedMemoryConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/CUDACvManagedMemoryConfig.cmake"
  INSTALL_DESTINATION "lib/cmake/CUDACvManagedMemory"
  PATH_VARS INCLUDE_INSTALL_DIR SYSCONFIG_INSTALL_DIR
)

# install the config file
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/CUDACvManagedMemoryConfig.cmake"
  DESTINATION "lib/cmake/CUDACvManagedMemory"
)



