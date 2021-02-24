cmake_minimum_required(VERSION 3.10)
project(tatooine)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules/")
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# preprocessor lib
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
set(TATOOINE_PP_NUM_MAX_INDICES "100" CACHE STRING "")
include(preprocessor_setup)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# git submodules
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
find_package(Git QUIET)
#if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  ## Update submodules as needed
  #execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                  #WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  #RESULT_VARIABLE GIT_SUBMOD_RESULT)
  #if(NOT GIT_SUBMOD_RESULT EQUAL "0")
    #message(FATAL_ERROR
      #"git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
  #endif()
#endif()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL
    "Enable building the unit tests which depend on gtest")
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL
    "Enable testing of the benchmark library.")
add_subdirectory(external/google-benchmark)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# configure files
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
set(TATOOINE_LIC_COMP_SHADER
    ${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/gpu/lic.comp)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/gpu/lic.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/include/tatooine/gpu/lic.h")

## Line Shader
set(TATOOINE_LINE_SHADER_VERT_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/gpu/line.vert)
set(TATOOINE_LINE_SHADER_GEOM_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/gpu/line.geom)
set(TATOOINE_LINE_SHADER_FRAG_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/gpu/line.frag)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/gpu/line_shader.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/include/tatooine/gpu/line_shader.h")

 ## Texture Shader
set(TATOOINE_TEXTURE_SHADER_VERT_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/gpu/texture.vert)
set(TATOOINE_TEXTURE_SHADER_FRAG_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/gpu/texture.frag)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/gpu/texture_shader.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/include/tatooine/gpu/texture_shader.h")

set(TATOOINE_GPU_REDUCE_TEX2R32F_COMP_SHADER_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/gpu/reduce_tex2r32f.comp)
set(TATOOINE_GPU_REDUCE_SHADERSTORAGE_COMP_SHADER_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/gpu/reduce_shaderstoragebuffer.comp)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/gpu/reduce.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/include/tatooine/gpu/reduce.h")
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# clang-tidy
set(TATOOINE_USE_CLANG_TIDY false CACHE BOOL
    "use clang-tidy checks while building")
option(TATOOINE_USE_CLANG_TIDY
       "use clang-tidy checks while building" FALSE)
find_program(CLANG_TIDY_EXECUTABLE NAMES "clang-tidy" DOC "Path to clang-tidy executable")
string(CONCAT TATOOINE_CLANG_TIDY_CHECKS
  "-checks="
  ",cppcoreguidelines*"
  ",-cppcoreguidelines-pro-type-reinterpret-cast"
  ",-cppcoreguidelines-pro-bounds-pointer-arithmetic"
  ",-cppcoreguidelines-pro-bounds-constant-array-index"
  ",-cppcoreguidelines-avoid-magic-numbers"
  ",-cppcoreguidelines-macro-usage"
  ",cert*"
  ",-cert-err34-c"
  ",hicpp"
  ",modernize*"
  ",performance*"
  ",readability*"
  ",-readability-magic-numbers"
  ",-readability-isolate-declaration"
  ",-readability-static-accessed-through-instance"
  ",-readability-named-parameter"
  ",google-explicit-constructor"
  )
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# create library target
add_library(tatooine SHARED 
  src/vtk_legacy.cpp
  src/parse.cpp
)
target_compile_features(tatooine PUBLIC cxx_std_17)

# force coloring
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  target_compile_options(tatooine PUBLIC "-fcolor-diagnostics")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  target_compile_options(tatooine PUBLIC  "-fdiagnostics-color=always")
endif()

#enable warnings
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR
    "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
target_compile_options(tatooine PRIVATE
  -Wall -Wextra -pedantic
  -Wno-tautological-undefined-compare -Wno-gnu-zero-variadic-macro-arguments)
endif()

try_compile(TATOOINE_STD_FILESYSTEM_AVAILABLE "${CMAKE_BINARY_DIR}/temp" 
            "${CMAKE_CURRENT_SOURCE_DIR}/src/test_has_filesystem.cpp" 
            CMAKE_FLAGS -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_STANDARD_REQUIRED=ON
            LINK_LIBRARIES stdc++fs)

#===============================================================================
# Required Packages
#===============================================================================
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(LAPACKE REQUIRED)
if(TATOOINE_STD_FILESYSTEM_AVAILABLE)
  find_package(Boost REQUIRED COMPONENTS program_options)
else()
  find_package(Boost REQUIRED COMPONENTS program_options filesystem)
endif()
find_package(vcode REQUIRED)

option(TATOOINE_INCLUDE_MKL_LAPACKE "If ON mkl_lapacke.h will be include instead of lapacke.h")
if (TATOOINE_INCLUDE_MKL_LAPACKE)
  target_compile_definitions(tatooine PUBLIC TATOOINE_INCLUDE_MKL_LAPACKE)
endif()

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# set include dirs
target_include_directories(tatooine
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/single_include
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/indicators/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/tomlplusplus/include
    ${LAPACKE_INCLUDE_DIRS}
    ${vcode_DIR}/../..
    ${Boost_INCLUDE_DIRS}
/lrz/sys/intel/studio2019_u5/compilers_and_libraries_2019.5.281/linux/mkl/include
)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#set libs 
target_link_libraries(tatooine
  PUBLIC
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    ${LAPACKE_LIBRARIES}
    ${Boost_PROGRAM_OPTIONS_LIBRARY}
)

if (TATOOINE_STD_FILESYSTEM_AVAILABLE)
  target_link_libraries(tatooine PUBLIC stdc++fs)
  target_compile_definitions(tatooine PUBLIC TATOOINE_STD_FILESYSTEM_AVAILABLE)
else()
target_link_libraries(tatooine PUBLIC
  ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY})
endif()
#===============================================================================
# Optional Packages
#===============================================================================
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# PNG
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
find_package(PNG QUIET)
if(PNG_FOUND)
  find_package(PNG++ QUIET)
  if (DEFINED PNG++_INCLUDE_DIRS)
    target_compile_definitions(tatooine PUBLIC TATOOINE_HAS_PNG_SUPPORT)
    target_include_directories(tatooine PUBLIC ${PNG_INCLUDE_DIRS} ${PNG++_INCLUDE_DIRS})
    target_link_libraries(tatooine PUBLIC ${PNG_LIBRARIES} ${PNG++_LIBRARIES})
    message(STATUS "Compiling with PNG support")
  endif()
endif()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# OpenMP
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
find_package(OpenMP QUIET)
if (OpenMP_C_FOUND)
  option(TATOOINE_COMPILE_WITH_OPENMP "" ON)
  if (TATOOINE_COMPILE_WITH_OPENMP)
    set(TATOOINE_OPENMP_AVAILABLE 1)
    target_link_libraries(tatooine PUBLIC ${OpenMP_C_LIBRARY} ${OpenMP_C_LIBRARIES} ${OpenMP_CXX_LIBRARY} ${OpenMP_CXX_LIBRARIES})
    target_compile_options(tatooine PUBLIC ${OpenMP_C_FLAGS} ${OpenMP_C_FLAGS} ${OpenMP_CXX_FLAGS} ${OpenMP_CXX_FLAGS})
    message(STATUS "Compiling with OpenMP")
  else()
    set(TATOOINE_OPENMP_AVAILABLE 0)
    message(STATUS "OpenMP not available")
  endif()
endif()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# NetCDF
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
find_package(NetCDF QUIET)
if (NETCDF_FOUND)
  message(STATUS "NetCDF available")
  target_compile_definitions(tatooine PUBLIC TATOOINE_HAS_NETCDF_SUPPORT)
  target_link_libraries(tatooine PUBLIC ${NETCDF_LIBRARIES})
  target_include_directories(tatooine PUBLIC ${NETCDF_INCLUDES})
endif()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# HDF5
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
find_package(HDF5 QUIET COMPONENTS CXX HL)
if(HDF5_FOUND)
  if (HDF5_IS_PARALLEL)
    message(STATUS "HDF5 available with parallel support")
  else()
    message(STATUS "HDF5 available without parallel support")
  endif()
  target_compile_definitions(tatooine PUBLIC
    TATOOINE_HAS_HDF5_SUPPORT ${HDF5_DEFINITIONS} ${HDF5_CXX_DEFINITIONS})
  target_link_libraries(tatooine PUBLIC
    ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES} ${HDF5_CXX_LIBRARIES} ${HDF5_CXX_HL_LIBRARIES})
  target_include_directories(tatooine PUBLIC
    ${HDF5_INCLUDE_DIRS} ${HDF5_CXX_INCLUDE_DIRS})

  #add_executable(hdf5_create src/hdf5_create.cpp)
  #target_link_libraries(hdf5_create PUBLIC tatooine)
  #add_custom_target(hdf5_create.run COMMAND ./hdf5_create DEPENDS hdf5_create)

  #add_executable(hdf5_readdata src/hdf5_readdata.cpp)
  #target_link_libraries(hdf5_readdata PUBLIC tatooine)
  #add_custom_target(hdf5_readdata.run COMMAND ./hdf5_readdata DEPENDS hdf5_readdata)
else()
  message(STATUS "HDF5 not available")
endif()

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# FLANN
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
find_package(FLANN QUIET)
if (FLANN_FOUND)
  target_compile_definitions(tatooine PUBLIC TATOOINE_HAS_FLANN_SUPPORT)
  target_link_libraries(tatooine PUBLIC FLANN::FLANN)
  message(STATUS "FLANN available")
endif()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# CGAL
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
set(CGAL_DO_NOT_WARN_ABOUT_CMAKE_BUILD_TYPE TRUE)
find_package(CGAL QUIET)
if (CGAL_FOUND)
  target_compile_definitions(tatooine PUBLIC TATOOINE_HAS_CGAL_SUPPORT)
  target_link_libraries(tatooine PUBLIC CGAL::CGAL)
  message(STATUS "CGAL available")
endif()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# yavin
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
find_package(yavin QUIET)
if (${yavin_FOUND}) 
  set(TATOOINE_YAVIN_AVAILABLE 1)
  target_link_libraries(tatooine PUBLIC yavin)
else()
  set(TATOOINE_YAVIN_AVAILABLE 0)
endif()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# GINAC
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
find_package(GINAC QUIET)
if (${GINAC_FOUND}) 
  option(TATOOINE_USE_GINAC "use GiNaC for symbolics support" FALSE)
  if (${TATOOINE_USE_GINAC})
    set(TATOOINE_GINAC_AVAILABLE 1)
    target_include_directories(tatooine PUBLIC ${GINAC_INCLUDE_DIR})
    target_link_libraries(tatooine PUBLIC ${GINAC_LIBRARY})
  else()
    set(TATOOINE_GINAC_AVAILABLE 0)
  endif()
else()
  set(TATOOINE_GINAC_AVAILABLE 0)
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/packages.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/include/tatooine/packages.h)

set(TATOOINE_EXPORTING_TARGETS tatooine)
#-------------------------------------------------------------------------------
# Subdirectories
#-------------------------------------------------------------------------------
add_subdirectory(test)
add_subdirectory(scivis_contest_2020)
add_subdirectory(benchmarks)
add_subdirectory(doc)
add_subdirectory(material_science)
if (yavin_FOUND) 
  add_subdirectory(steadification)
endif()
option(TATOOINE_BUILD_FLOWEXPLORER "Builds Flow Explorer" OFF)
if (TATOOINE_BUILD_FLOWEXPLORER AND yavin_FOUND) 
  add_subdirectory(steadification)
endif()
add_subdirectory(autonomous_particles)
add_subdirectory(smearing_scalar_field)

find_package(MPI COMPONENTS C CXX)
if (MPI_C_FOUND)
  option(TATOOINE_BUILD_MPI "Builds MPI-related projecs")
  if (TATOOINE_BUILD_MPI)
    add_subdirectory(mpi)
    add_subdirectory(insitu)
    add_subdirectory(channelflow)
  endif()
endif()

option(TATOOINE_BUILD_EXAMPLES "")
if (TATOOINE_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# This makes the project importable from the build directory
export(TARGETS ${TATOOINE_EXPORTING_TARGETS}
       FILE tatooineConfig.cmake
       NAMESPACE tatooine::)
export(PACKAGE tatooine)
