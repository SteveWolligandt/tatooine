cmake_minimum_required(VERSION 3.10)
project(tatooine)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules/")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# preprocessor lib
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
set(TATOOINE_SOURCE_DIR ${CMAKE_SOURCE_DIR})
set(TATOOINE_CURRENT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(TATOOINE_BINARY_DIR ${CMAKE_BINARY_DIR})
set(TATOOINE_CURRENT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(TATOOINE_EXTERNAL ${CMAKE_SOURCE_DIR}/external)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# preprocessor lib
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
set(TATOOINE_PP_NUM_MAX_INDICES "100" CACHE STRING "")
include(preprocessor_setup)
include(target_link_libraries_system)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# git submodules
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
find_package(Git REQUIRED)
function(update_git_submodule dir)
  execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive -- ${dir}
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  RESULT_VARIABLE GIT_SUBMOD_RESULT)
  if(NOT GIT_SUBMOD_RESULT EQUAL "0")
    message(FATAL_ERROR
      "When updating submodule ${dir}:\n  git submodule update --init --recursive -- ${dir}\nfailed with ${GIT_SUBMOD_RESULT}, please checkout submodules.")
  endif()
endfunction()
set(TATOOINE_MANDATORY_GIT_SUBMODULES indicators)
foreach(SUBMODULE ${TATOOINE_MANDATORY_GIT_SUBMODULES})
  update_git_submodule(${TATOOINE_EXTERNAL}/${SUBMODULE})
endforeach()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL
    "Enable installation of benchmark. (Projects embedding benchmark may want to turn this OFF.)")
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL
    "Enable building the unit tests which depend on gtest")
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL
    "Enable testing of the benchmark library.")
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
find_program(CLANG_TIDY_EXECUTABLE NAMES "clang-tidy" DOC "Path to clang-tidy executable")
if (CLANG_TIDY_EXECUTABLE)
  option(TATOOINE_USE_CLANG_TIDY
         "use clang-tidy checks while building" FALSE)
  string(CONCAT CLANG_TIDY_CHECKS
    "-*"
    ",cppcoreguidelines*"
    ",hicpp-*"
    ",modernize-*"
    ",performance-*"
    ",readability-*"
    ",-cppcoreguidelines-pro-type-reinterpret-cast"
    ",-cppcoreguidelines-pro-bounds-pointer-arithmetic"
    ",-cppcoreguidelines-pro-bounds-constant-array-index"
    ",-cppcoreguidelines-macro-usage"
    ",-cert-err34-c"
    ",-readability-magic-numbers"
    ",-readability-isolate-declaration"
    ",-readability-static-accessed-through-instance"
    ",-readability-named-parameter"
    )
  set(CLANG_TIDY_DOMMAND
    ${CLANG_TIDY_EXECUTABLE}
    -header-filter=${CMAKE_SOURCE_DIR}
    -checks=-*,cppcoreguidelines-*,modernize-*)
else()
  set(TATOOINE_USE_CLANG_TIDY FALSE)
endif()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# create library target
add_library(tatooine SHARED 
  src/vtk_legacy.cpp
  src/parse.cpp
)
target_compile_options(tatooine PUBLIC -Wno-inconsistent-missing-override)
target_compile_features(tatooine PUBLIC cxx_std_20)

# force coloring
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  target_compile_options(tatooine PUBLIC -fcolor-diagnostics)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  target_compile_options(tatooine PUBLIC -fdiagnostics-color=always)
endif()

#enable warnings
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  target_compile_options(tatooine PUBLIC
    -Wall -Wextra -pedantic -Wno-cpp)
endif()
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  target_compile_options(tatooine PUBLIC
    -Wall -Wextra -pedantic -Wno-cpp)
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
target_compile_definitions(tatooine PUBLIC BOOST_BIND_GLOBAL_PLACEHOLDERS)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# check if mkl_lapacke.h can be used
get_target_property(TATOOINE_INCLUDE_DIRS tatooine INCLUDE_DIRECTORIES)
set(TATOOINE_INCLUDE_MKL_LAPACKE OFF)
foreach(dir ${TATOOINE_INCLUDE_DIRS})
  if(EXISTS "${dir}/mkl_lapacke.h")
    set(TATOOINE_INCLUDE_MKL_LAPACKE ON)
    break()
  endif()
endforeach()
if (TATOOINE_INCLUDE_MKL_LAPACKE)
  target_compile_definitions(tatooine PUBLIC TATOOINE_INCLUDE_MKL_LAPACKE)
endif()
unset(TATOOINE_INCLUDE_DIRS)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# set include dirs
target_include_directories(tatooine
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/single_include
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/indicators/include
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
    set(TATOOINE_PNG_AVAILABLE 1)
  else()
    set(TATOOINE_PNG_AVAILABLE 0)
  endif()
else()
  set(TATOOINE_PNG_AVAILABLE 0)
endif()
target_compile_definitions(tatooine PUBLIC TATOOINE_PNG_AVAILABLE=${TATOOINE_PNG_AVAILABLE})
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
    target_compile_definitions(tatooine PUBLIC TATOOINE_OPENMP_AVAILABLE)
    message(STATUS "Compiling with OpenMP")
  else()
    set(TATOOINE_OPENMP_AVAILABLE 0)
    message(STATUS "OpenMP not available")
  endif()
endif()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# MPI
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
get_filename_component(C_COMPILER_EXECUTABLE ${CMAKE_C_COMPILER} NAME)
get_filename_component(CXX_COMPILER_EXECUTABLE ${CMAKE_CXX_COMPILER} NAME)
if ("${CXX_COMPILER_EXECUTABLE}" STREQUAL "mpicxx")
  find_package(MPI COMPONENTS C QUIET)
  if (MPI_C_FOUND)
    option(TATOOINE_BUILD_MPI "Builds MPI-related projecs" ON)
    if (TATOOINE_BUILD_MPI)
      set(TATOOINE_MPI_AVAILABLE 1)
    else()
      set(TATOOINE_MPI_AVAILABLE 0)
    endif()
  else()
    set(TATOOINE_MPI_AVAILABLE 0)
  endif()
else()
  message(STATUS "MPI can only be used if CXX_COMPILER_EXECUTABLE is mpicxx. Current: ${CXX_COMPILER_EXECUTABLE}")
  set(TATOOINE_MPI_AVAILABLE 0)
endif()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# HDF5
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
option(TATOOINE_USE_SYSTEM_HDF5_LIBS "By setting this to ON tatooine will not compile HDF5 on its own." OFF)
if (TATOOINE_USE_SYSTEM_HDF5_LIBS) 
  find_package(HDF5 QUIET COMPONENTS C)
  if(HDF5_FOUND)
    if (HDF5_IS_PARALLEL)
      message(STATUS "HDF5 available with MPI support")
    else()
      message(STATUS "HDF5 available without MPI support")
    endif()
    target_compile_definitions(tatooine PUBLIC
      TATOOINE_HDF5_AVAILABLE ${HDF5_DEFINITIONS} ${HDF5_C_DEFINITIONS})
    target_link_libraries(tatooine PUBLIC
      ${HDF5_LIBRARIES})
    target_include_directories(tatooine PUBLIC
      ${HDF5_INCLUDE_DIRS})
  else()
    message(STATUS "HDF5 not available")
  endif()
  set(TATOOINE_HDF5_HEADER hdf5.h)
endif()
if ((NOT TATOOINE_USE_SYSTEM_HDF5_LIBS) OR
    (TATOOINE_USE_SYSTEM_HDF5_LIBS AND NOT HDF5_FOUND))
  update_git_submodule(${TATOOINE_EXTERNAL}/hdf5)
  set(HDF5_EXTERNALLY_CONFIGURED 1)
  set(HDF5_EXTERNAL_LIB_PREFIX "tatooine" CACHE STRING "Use prefix for custom library naming.")
  set(HDF5_BUILD_CPP_LIB     OFF CACHE BOOL "Build HDF5 C++ Library")
  set(HDF5_BUILD_HL_LIB      OFF CACHE BOOL "Build HIGH Level HDF5 Library")
  set(HDF5_BUILD_JAVA        OFF CACHE BOOL "Build Java HDF5 Library")
  set(HDF5_BUILD_TOOLS       OFF CACHE BOOL "Build HDF5 Tools")
  set(HDF5_BUILD_UTILS       OFF CACHE BOOL "Build HDF5 Utils")
  set(HDF5_BUILD_EXAMPLES    OFF CACHE BOOL "Build HDF5 Library Examples")
  set(HDF5_ENABLE_COVERAGE   OFF CACHE BOOL "Enable code coverage for Libraries and Programs")
  set(HDF5_ENABLE_THREADSAFE ON  CACHE BOOL "Enable thread-safety")
  if (("${C_COMPILER_EXECUTABLE}" STREQUAL "mpicc" OR
       "${C_COMPILER_EXECUTABLE}" STREQUAL "mpiicc") AND TATOOINE_MPI_AVAILABLE)
    message(STATUS "Building HDF5 with thread-safety and MPI support.")
    set(ALLOW_UNSUPPORTED ON)
    set(HDF5_ENABLE_PARALLEL ON CACHE BOOL "Enable parallel build (requires MPI)")
    set(HDF5_ENABLE_PARALLEL ON)
  else()
    message(STATUS "Building HDF5 with thread-safety but without MPI support.")
    set(HDF5_ENABLE_PARALLEL OFF CACHE BOOL "Enable parallel build (requires MPI)")
    set(HDF5_ENABLE_PARALLEL OFF)
  endif()
  add_subdirectory(external/hdf5 EXCLUDE_FROM_ALL)
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR
      "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(TATOOINE_HDF5_WNOS
      -Wno-larger-than
      -Wno-null-dereference
      -Wno-strict-overflow
      -Wno-missing-field-initializers
      -Wno-cast-qual
      -Wno-sign-conversion
      -Wno-unused-function
      -Wno-unused-variable
      -Wno-unused-parameter
      -Wno-unused-label)
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
      list(APPEND TATOOINE_HDF5_WNOS
        -Wno-stringop-overflow
        -Wno-maybe-uninitialized
        -Wno-discarded-qualifiers
        -Wno-cast-function-type
        -Wno-stringop-truncation)
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
      list(APPEND TATOOINE_HDF5_WNOS
        -Wno-unknown-warning-option
        -Wno-double-promotion
        -Wno-discarded-qualifiers
        -Wno-misleading-indentation
        -Wno-incompatible-pointer-types-discards-qualifiers
        -Wno-format-pedantic
        -Wno-implicit-fallthrough
        -Wno-cast-function-type
        -Wno-maybe-uninitialized
        -Wno-stringop-truncation
        -Wno-stringop-overflow)
    endif()
    target_compile_options(hdf5-static PRIVATE ${TATOOINE_HDF5_WNOS})
  endif()


  target_link_libraries(tatooine PUBLIC hdf5-static)
  target_compile_definitions(tatooine PUBLIC TATOOINE_HDF5_AVAILABLE)
  list(APPEND TATOOINE_EXPORTING_TARGETS hdf5-static)
  target_compile_definitions(tatooine PUBLIC TATOOINE_USE_SYSTEM_HDF5_LIBS=0)
  set(TATOOINE_HDF5_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/external/hdf5/src/hdf5.h)
endif()
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/hdf5.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/include/tatooine/hdf5.h)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# NetCDF
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if(TATOOINE_USE_SYSTEM_HDF5_LIBS)
  option(TATOOINE_USE_SYSTEM_NETCDF_LIBS "By setting this to ON tatooine will not compile NetCDF on its own." OFF)
  if (TATOOINE_USE_SYSTEM_NETCDF_LIBS)
    find_package(NetCDF QUIET)
    if (NETCDF_FOUND)
      message(STATUS "NetCDF available")
      set(TATOOINE_NETCDF_AVAILABLE 1)
      target_compile_definitions(tatooine PUBLIC TATOOINE_NETCDF_AVAILABLE=1)
      target_link_libraries(tatooine PUBLIC ${NETCDF_LIBRARIES})
      target_include_directories(tatooine PUBLIC ${NETCDF_INCLUDES})
    endif()
  endif()
endif()
if ((NOT TATOOINE_USE_SYSTEM_NETCDF_LIBS) OR
  (TATOOINE_USE_SYSTEM_NETCDF_LIBS AND NOT NETCDF_FOUND))
  set(TATOOINE_NETCDF_AVAILABLE 0)
  target_compile_definitions(tatooine PUBLIC TATOOINE_NETCDF_AVAILABLE=0)
  #update_git_submodule(${TATOOINE_EXTERNAL}/netcdf-c)
  #update_git_submodule(${TATOOINE_EXTERNAL}/netcdf-cxx4)
  #add_subdirectory(external/netcdf-c EXCLUDE_FROM_ALL)
  #add_subdirectory(external/netcdf-cxx4 EXCLUDE_FROM_ALL)
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
  target_compile_definitions(tatooine PUBLIC -DCGAL_DISABLE_ROUNDING_MATH_CHECK)
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
get_target_property(defs tatooine COMPILE_DEFINITIONS)
list(REMOVE_ITEM defs _FORTIFY_SOURCE)
set_property(TARGET tatooine PROPERTY COMPILE_DEFINITIONS ${defs})

set(TATOOINE_EXPORTING_TARGETS tatooine)
#-------------------------------------------------------------------------------
# Subdirectories
#-------------------------------------------------------------------------------
add_subdirectory(gl)
target_compile_definitions(tatooine PUBLIC TATOOINE_GL_AVAILABLE=${TATOOINE_GL_AVAILABLE})
add_subdirectory(test)
add_subdirectory(scivis_contest_2020 EXCLUDE_FROM_ALL)
add_subdirectory(benchmarks)
option(TATOOINE_BUILD_DOC "Builds the documentation" OFF)
if (TATOOINE_BUILD_DOC) 
  add_subdirectory(doc)
endif () 
add_subdirectory(material_science EXCLUDE_FROM_ALL)
if (TATOOINE_GL_AVAILABLE)
  #add_subdirectory(steadification EXCLUDE_FROM_ALL)

  option(TATOOINE_BUILD_FLOWEXPLORER "Builds Flow Explorer" OFF)
  if (TATOOINE_BUILD_FLOWEXPLORER)
    add_subdirectory(flowexplorer)
  endif()
endif()
add_subdirectory(raytracer)
add_subdirectory(autonomous_particles)
add_subdirectory(smearing_scalar_field)

if (TATOOINE_MPI_AVAILABLE)
  add_subdirectory(mpi)
  add_subdirectory(insitu)
endif()

option(TATOOINE_BUILD_CHANNELFLOW_VISUALIZATION
       "creates targets to make visualizations for the channelflow" ON)
if(TATOOINE_BUILD_CHANNELFLOW_VISUALIZATION)
  add_subdirectory(channelflow)
endif()

option(TATOOINE_BUILD_EXAMPLES "")
if (TATOOINE_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
add_subdirectory(doktorandentag)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# This makes the project importable from the build directory
#export(TARGETS ${TATOOINE_EXPORTING_TARGETS}
#       FILE tatooineConfig.cmake
#       NAMESPACE tatooine::)
export(PACKAGE tatooine)
