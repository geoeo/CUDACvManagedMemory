#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "CUDACvManagedMemory" for configuration ""
set_property(TARGET CUDACvManagedMemory APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(CUDACvManagedMemory PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libCUDACvManagedMemory.so"
  IMPORTED_SONAME_NOCONFIG "libCUDACvManagedMemory.so"
  )

list(APPEND _cmake_import_check_targets CUDACvManagedMemory )
list(APPEND _cmake_import_check_files_for_CUDACvManagedMemory "${_IMPORT_PREFIX}/lib/libCUDACvManagedMemory.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
