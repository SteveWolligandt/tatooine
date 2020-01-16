cmake_minimum_required(VERSION 3.10)
project(tatooine)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules/")

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
find_package(GiNaC REQUIRED)
find_package(LAPACKE REQUIRED)
find_package(OpenMP REQUIRED)
find_package(vcode REQUIRED)
find_package(PNG REQUIRED)
find_package(PNG++ REQUIRED)
find_package(HDF5 REQUIRED)
set(HDF5_LIBS -lhdf5 -lhdf5_cpp -lhdf5_hl)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# create library target
add_library(tatooine INTERFACE)
target_compile_features(tatooine INTERFACE cxx_std_17)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# set include dirs
target_include_directories(tatooine
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/single_include
    ${GINAC_INCLUDE_DIRS}
    ${LAPACKE_INCLUDE_DIR}
    ${vcode_DIR}/../..
    ${PNG_INCLUDE_DIRS}
    ${PNG++_INCLUDE_DIRS}
)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#set libs 
target_link_libraries(tatooine
  INTERFACE
    ${GINAC_LIBRARIES}
    ${LAPACKE_LIBRARIES}
    ${PNG_LIBRARIES}
    ${PNG++_LIBRARIES}
    ${HDF5_LIBS}
)
if(OpenMP_CXX_FOUND)
  target_link_libraries(tatooine INTERFACE OpenMP::OpenMP_CXX)
endif()

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
add_subdirectory(test)
add_subdirectory(misc)
add_subdirectory(cuda)
add_subdirectory(steadification)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# This makes the project importable from the build directory
export(TARGETS tatooine FILE tatooineConfig.cmake)
export(PACKAGE tatooine)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Doxygen
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# check if Doxygen is installed
find_package(Doxygen)
if (DOXYGEN_FOUND)
  # first we can indicate the documentation build as an option and set it to ON by default
  option(BUILD_DOC "Generate documentation" OFF)
  if (BUILD_DOC)
    # set input and output files
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/doc/Doxyfile)

    # request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    # note the option ALL which allows to build the docs together with the application
    add_custom_target(doc ALL
      COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Generating API documentation with Doxygen"
      VERBATIM)
  endif()
endif ()
