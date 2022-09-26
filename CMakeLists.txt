cmake_minimum_required(VERSION 3.15)
project(Tatooine)
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules/")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(FetchContent)
#--------------------------------------------------------------------------------
# git submodules
#--------------------------------------------------------------------------------
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
option(TATOOINE_BUILD_DOC "Builds the documentation" OFF)
if (TATOOINE_BUILD_DOC) 
  option(TATOOINE_DOC_ONLY "Only builds documentation" OFF)
  add_subdirectory(doc)
  target_compile_options(base INTERFACE -DTATOOINE_DOC_ONLY)
endif () 
#--------------------------------------------------------------------------------
set(TATOOINE_MANDATORY_GIT_SUBMODULES
  external/rapidxml
  external/indicators
  external/gcem)
foreach(SUBMODULE ${TATOOINE_MANDATORY_GIT_SUBMODULES})
  update_git_submodule(${SUBMODULE})
endforeach()
#--------------------------------------------------------------------------------
set(REAL_NUMBER double float)
set(TATOOINE_REAL_NUMBER_TYPE double CACHE STRING "Floating point type for real numbers")
set_property(CACHE TATOOINE_REAL_NUMBER_TYPE PROPERTY STRINGS ${REAL_NUMBER})
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/real.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/tatooine/real.h)

#--------------------------------------------------------------------------------
# filesystem check
#--------------------------------------------------------------------------------
if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
  try_compile(TATOOINE_STD_FILESYSTEM_COMPILED "${CMAKE_BINARY_DIR}/temp" 
              "${CMAKE_CURRENT_SOURCE_DIR}/src/test_has_filesystem.cpp" 
              CMAKE_FLAGS -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_STANDARD_REQUIRED=ON
              LINK_LIBRARIES stdc++fs)
elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "Windows")
  try_compile(TATOOINE_STD_FILESYSTEM_COMPILED "${CMAKE_BINARY_DIR}/temp" 
              "${CMAKE_CURRENT_SOURCE_DIR}/src/test_has_filesystem.cpp" 
              CMAKE_FLAGS -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_STANDARD_REQUIRED=ON)
endif()
#--------------------------------------------------------------------------------
set(TATOOINE_EXTERNAL ${CMAKE_SOURCE_DIR}/external)
include(target_link_libraries_system)
#-------------------------------------------------------------------------------
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
#-------------------------------------------------------------------------------
# create library target
add_library(base INTERFACE)

if("${CMAKE_SYSTEM_NAME}" STREQUAL "Windows")
  target_compile_definitions(base INTERFACE _USE_MATH_DEFINES)
  target_compile_definitions(base INTERFACE NOMINMAX)
endif()
set_property(TARGET base PROPERTY CXX_STANDARD 20)
#-------------------------------------------------------------------------------
# Boost
#-------------------------------------------------------------------------------
find_package(Boost REQUIRED)
target_link_libraries(base INTERFACE Boost::headers)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR
    "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  target_compile_options(base INTERFACE -Wno-interference-size)
endif()
#-------------------------------------------------------------------------------
# set include dirs
target_include_directories(base
  INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
    external/indicators/include
    external/vcode/include
    external/gcem/include
    external/rapidxml)

if (TATOOINE_STD_FILESYSTEM_COMPILED)
  set(TATOOINE_STD_FILESYSTEM_AVAILABLE 1)
  if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
    target_link_libraries(base INTERFACE stdc++fs)
  endif()
else()
  set(TATOOINE_STD_FILESYSTEM_AVAILABLE 0)
  find_package(Boost REQUIRED COMPONENTS filesystem)
  target_link_libraries(base INTERFACE Boost::filesystem)
endif()
target_compile_definitions(base
 INTERFACE
  TATOOINE_STD_FILESYSTEM_AVAILABLE=${TATOOINE_STD_FILESYSTEM_AVAILABLE})

# force coloring
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  target_compile_options(base INTERFACE -fcolor-diagnostics)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  target_compile_options(base INTERFACE -fdiagnostics-color=always)
endif()

#enable warnings
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR
    "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  target_compile_options(base INTERFACE -Wall -Wextra -Wpedantic)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  target_compile_options(base INTERFACE /W3)
endif()
#-------------------------------------------------------------------------------
# force coloring
#-------------------------------------------------------------------------------
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  target_compile_options(base INTERFACE -fcolor-diagnostics)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  target_compile_options(base INTERFACE -fdiagnostics-color=always)
endif()

if (NOT TATOOINE_DOC_ONLY)
#===============================================================================
# Required Packages
#===============================================================================
#-------------------------------------------------------------------------------
# OpenMP
#-------------------------------------------------------------------------------
find_package(OpenMP REQUIRED COMPONENTS CXX)
if(OpenMP_FOUND)
  option(TATOOINE_COMPILE_WITH_OPENMP "" ON)
  if(TATOOINE_COMPILE_WITH_OPENMP)
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
      target_compile_options(base INTERFACE "/openmp:llvm")
    else()
    target_link_libraries(base INTERFACE OpenMP::OpenMP_CXX)
    endif()
    set(TATOOINE_OPENMP_AVAILABLE 1)
    message(STATUS "Compiling with OpenMP")
  else()
    set(TATOOINE_OPENMP_AVAILABLE 0)
    message(STATUS "OpenMP not available")
  endif()
endif()
target_compile_definitions(base INTERFACE TATOOINE_OPENMP_AVAILABLE=${TATOOINE_OPENMP_AVAILABLE})

#===============================================================================
# Optional Packages
#===============================================================================

#-------------------------------------------------------------------------------
# PNG
#-------------------------------------------------------------------------------
find_package(PNG)
if(PNG_FOUND)
  find_package(PNG++)
  if (DEFINED PNG++_INCLUDE_DIRS)
    target_include_directories(base INTERFACE ${PNG_INCLUDE_DIRS} ${PNG++_INCLUDE_DIRS})
    target_link_libraries(base INTERFACE ${PNG_LIBRARIES} ${PNG++_LIBRARIES})
    message(STATUS "${PNG_LIBRARIES} ${PNG++_LIBRARIES}")
    set(TATOOINE_PNG_AVAILABLE 1)
  else()
    set(TATOOINE_PNG_AVAILABLE 0)
  endif()
else()
  set(TATOOINE_PNG_AVAILABLE 0)
endif()
target_compile_definitions(base INTERFACE TATOOINE_PNG_AVAILABLE=${TATOOINE_PNG_AVAILABLE})
#-------------------------------------------------------------------------------
# MPI
#-------------------------------------------------------------------------------
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
#-------------------------------------------------------------------------------
# HDF5
#-------------------------------------------------------------------------------
option(TATOOINE_USE_HDF5 "Compile with HDF5 support" ON)
if (TATOOINE_USE_HDF5)
  find_package(HDF5 QUIET COMPONENTS C)
  if(HDF5_FOUND)
    option(TATOOINE_USE_SYSTEM_HDF5_LIBS "By setting this to ON tatooine will not compile HDF5 on its own." ON)
  else()
    set(TATOOINE_USE_SYSTEM_HDF5_LIBS OFF)
  endif()
  if (TATOOINE_USE_SYSTEM_HDF5_LIBS) 
    if (HDF5_IS_PARALLEL)
      message(STATUS "HDF5 available with MPI support")
    else()
      message(STATUS "HDF5 available without MPI support")
    endif()
    target_compile_definitions(base INTERFACE
      ${HDF5_DEFINITIONS} ${HDF5_C_DEFINITIONS})
    set(TATOOINE_HDF5_AVAILABLE 1) 
    target_link_libraries(base INTERFACE
      ${HDF5_LIBRARIES})
    target_include_directories(base INTERFACE
      ${HDF5_INCLUDE_DIRS})
    set(TATOOINE_HDF5_HEADER hdf5.h)
  else ()
    update_git_submodule(${TATOOINE_EXTERNAL}/hdf5)
    set(ALLOW_UNSUPPORTED ON)
    set(HDF5_EXTERNALLY_CONFIGURED 1)
    set(HDF5_EXTERNAL_LIB_PREFIX "tatooine" CACHE STRING "Use prefix for custom library naming.")
    set(HDF5_BUILD_CPP_LIB             OFF CACHE BOOL "Build HDF5 C++ Library")
    set(HDF5_BUILD_HL_LIB              OFF CACHE BOOL "Build HIGH Level HDF5 Library")
    set(HDF5_BUILD_JAVA                OFF CACHE BOOL "Build Java HDF5 Library")
    set(HDF5_BUILD_TOOLS               OFF CACHE BOOL "Build HDF5 Tools")
    set(HDF5_BUILD_UTILS               OFF CACHE BOOL "Build HDF5 Utils")
    set(HDF5_BUILD_EXAMPLES            OFF CACHE BOOL "Build HDF5 Library Examples")
    set(HDF5_ENABLE_COVERAGE           OFF CACHE BOOL "Enable code coverage for Libraries and Programs")
    set(HDF5_ENABLE_THREADSAFE         ON  CACHE BOOL "Enable thread-safety")
    set(HDF5_ENABLE_DEPRECATED_SYMBOLS OFF CACHE BOOL "Enable deprecated public API symbols")
    if (("${C_COMPILER_EXECUTABLE}" STREQUAL "mpicc" OR
         "${C_COMPILER_EXECUTABLE}" STREQUAL "mpiicc") AND TATOOINE_MPI_AVAILABLE)
      message(STATUS "Building HDF5 with thread-safety and MPI support.")
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


    target_link_libraries(base INTERFACE hdf5-static)
    set(TATOOINE_HDF5_AVAILABLE 1) 
    list(APPEND TATOOINE_EXPORTING_TARGETS hdf5-static)
    target_compile_definitions(base INTERFACE TATOOINE_USE_SYSTEM_HDF5_LIBS=0)
    set(TATOOINE_HDF5_HEADER ${TATOOINE_EXTERNAL}/hdf5/src/hdf5.h)
  endif()
else()
  set(TATOOINE_HDF5_AVAILABLE 0) 
endif()
target_compile_definitions(base INTERFACE
  TATOOINE_HDF5_AVAILABLE=${TATOOINE_HDF5_AVAILABLE})
if (TATOOINE_HDF5_AVAILABLE)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/hdf5_include.h.in
                 ${CMAKE_CURRENT_BINARY_DIR}/include/tatooine/hdf5_include.h)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/hdf5/type.h.in
                 ${CMAKE_CURRENT_BINARY_DIR}/include/tatooine/hdf5/type.h)
endif()
#-------------------------------------------------------------------------------
# NetCDF
#-------------------------------------------------------------------------------
if(TATOOINE_USE_SYSTEM_HDF5_LIBS)
#  option(TATOOINE_USE_SYSTEM_NETCDF_LIBS
#         "By setting this to ON tatooine will not compile NetCDF on its own."
#         OFF)
#  if (TATOOINE_USE_SYSTEM_NETCDF_LIBS)
    find_package(NetCDF QUIET)
    if (NETCDF_FOUND)
      message(STATUS "NetCDF available")
      set(TATOOINE_NETCDF_AVAILABLE 1)
      target_compile_definitions(base INTERFACE TATOOINE_NETCDF_AVAILABLE=1)
      target_link_libraries(base INTERFACE ${NETCDF_LIBRARIES})
      target_include_directories(base INTERFACE ${NETCDF_INCLUDES})
    else()
      target_compile_definitions(base INTERFACE TATOOINE_NETCDF_AVAILABLE=0)
    endif()
#  endif()
#endif()
#if ((NOT TATOOINE_USE_SYSTEM_NETCDF_LIBS) OR
#  (TATOOINE_USE_SYSTEM_NETCDF_LIBS AND NOT NETCDF_FOUND))
#  set(TATOOINE_NETCDF_AVAILABLE 0)
#  target_compile_definitions(base INTERFACE TATOOINE_NETCDF_AVAILABLE=0)
#  #update_git_submodule(${TATOOINE_EXTERNAL}/netcdf-c)
#  #update_git_submodule(${TATOOINE_EXTERNAL}/netcdf-cxx4)
#  #add_subdirectory(external/netcdf-c EXCLUDE_FROM_ALL)
#  #add_subdirectory(external/netcdf-cxx4 EXCLUDE_FROM_ALL)
endif()

#-------------------------------------------------------------------------------
# GINAC
#-------------------------------------------------------------------------------
#find_package(GINAC QUIET)
#if (${GINAC_FOUND}) 
#  option(TATOOINE_USE_GINAC "use GiNaC for symbolics support" FALSE)
#  if (${TATOOINE_USE_GINAC})
#    set(TATOOINE_GINAC_AVAILABLE 1)
#    target_include_directories(base INTERFACE ${GINAC_INCLUDE_DIR})
#    target_link_libraries(base INTERFACE ${GINAC_LIBRARY})
#  else()
#    set(TATOOINE_GINAC_AVAILABLE 0)
#  endif()
#else()
  set(TATOOINE_GINAC_AVAILABLE 0)
#endif()

get_target_property(defs base COMPILE_DEFINITIONS)
list(REMOVE_ITEM defs _FORTIFY_SOURCE)
set_property(TARGET base PROPERTY COMPILE_DEFINITIONS ${defs})


#-------------------------------------------------------------------------------
# Subdirectories
#-------------------------------------------------------------------------------
add_subdirectory(preprocessor)
target_link_libraries(base INTERFACE preprocessor)
add_subdirectory(multidim_array)
add_subdirectory(tensor)
add_subdirectory(geometry)
add_subdirectory(fields)

if (TATOOINE_MPI_AVAILABLE)
  add_subdirectory(mpi)
  add_subdirectory(insitu)
endif()

add_subdirectory(gl)
add_subdirectory(rendering)

option(TATOOINE_BUILD_TESTS "" OFF)
if (TATOOINE_BUILD_TESTS)
  add_subdirectory(test)
endif()

option(TATOOINE_BUILD_BENCHMARKS "" OFF)
if (TATOOINE_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks EXCLUDE_FROM_ALL)
endif()

option(TATOOINE_BUILD_EXAMPLES "" OFF)
if (TATOOINE_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

#if (TATOOINE_GL_AVAILABLE)
#  message(STATUS "gl avalaible in main")
#  #add_subdirectory(steadification EXCLUDE_FROM_ALL)
#
#  option(TATOOINE_BUILD_FLOWEXPLORER "Builds Flow Explorer" OFF)
#  if (TATOOINE_BUILD_FLOWEXPLORER)
#    add_subdirectory(flowexplorer)
#  endif()
#endif()
endif()
#-------------------------------------------------------------------------------
configure_file(${CMAKE_SOURCE_DIR}/tatooineConfig.cmake.in ${CMAKE_BINARY_DIR}/tatooineConfig.cmake @ONLY)

export(
  TARGETS base geometry multidim_array tensor preprocessor fields
  APPEND FILE "${CMAKE_BINARY_DIR}/tatooineConfig.cmake"
  NAMESPACE tatooine::)

#if (TATOOINE_BLAS_AND_LAPACK_AVAILABLE) 
#  export(
#    TARGETS BLAS::BLAS LAPACK::LAPACK
#    APPEND FILE "${CMAKE_BINARY_DIR}/tatooineConfig.cmake"
#    NAMESPACE tatooine::)
#endif()
if (TATOOINE_GL_AVAILABLE) 
  export(
    TARGETS gl glfw imgui glad
    APPEND FILE "${CMAKE_BINARY_DIR}/tatooineConfig.cmake"
    NAMESPACE tatooine::)
endif()

export(
  TARGETS rendering
  APPEND FILE "${CMAKE_BINARY_DIR}/tatooineConfig.cmake"
  NAMESPACE tatooine::)

if(TATOOINE_CDT_AVAILABLE)
  export(
    TARGETS CDT
    APPEND FILE "${CMAKE_BINARY_DIR}/tatooineConfig.cmake"
    NAMESPACE tatooine::)
endif()

export(PACKAGE tatooine)
#--------------------------------------------------------------------------------
# configure files
#--------------------------------------------------------------------------------
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/available_libraries.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/include/tatooine/available_libraries.h")
