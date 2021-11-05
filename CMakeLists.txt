cmake_minimum_required(VERSION 3.10)
project(tatooine)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules/")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# filesystem check
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
try_compile(TATOOINE_STD_FILESYSTEM_AVAILABLE "${CMAKE_BINARY_DIR}/temp" 
            "${CMAKE_CURRENT_SOURCE_DIR}/src/test_has_filesystem.cpp" 
            CMAKE_FLAGS -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_STANDARD_REQUIRED=ON
            LINK_LIBRARIES stdc++fs)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
set(TATOOINE_EXTERNAL ${CMAKE_SOURCE_DIR}/external)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# preprocessor lib
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
set(TATOOINE_PP_NUM_MAX_INDICES "100" CACHE STRING "")
include(preprocessor_setup)
include(target_link_libraries_system)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# configure files
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/available_libraries.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/include/tatooine/available_libraries.h")
set(TATOOINE_LIC_COMP_SHADER
    ${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/gpu/lic.comp)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/gpu/lic.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/include/tatooine/gpu/lic.h")

## Line Shader
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/gpu/line_shader.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/include/tatooine/gpu/line_shader.h")

 ## Texture Shader
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/gpu/texture_shader.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/include/tatooine/gpu/texture_shader.h")

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
set(TATOOINE_SRC_FILES 
  src/vtk_legacy.cpp
  src/parse.cpp
)
# create library target
add_library(tatooine STATIC ${TATOOINE_SRC_FILES})
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# set include dirs
target_include_directories(tatooine
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
    external/indicators/include
    external/vcode/include)

if (TATOOINE_STD_FILESYSTEM_AVAILABLE)
  target_link_libraries(tatooine PUBLIC stdc++fs)
  target_compile_definitions(tatooine PUBLIC TATOOINE_STD_FILESYSTEM_AVAILABLE)
endif()

set(TATOOINE_EXPORTING_TARGETS tatooine)
target_compile_features(tatooine PUBLIC cxx_std_20)

# force coloring
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  target_compile_options(tatooine PUBLIC -fcolor-diagnostics)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  target_compile_options(tatooine PUBLIC -fdiagnostics-color=always)
endif()

#enable warnings
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR
    "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  target_compile_options(tatooine PUBLIC
    -Wall -Wextra -Wpedantic
    -Wno-cpp
    -Wno-inconsistent-missing-override
    -Wno-c99-extensions)
endif()



#-------------------------------------------------------------------------------
# force coloring
#-------------------------------------------------------------------------------
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  target_compile_options(tatooine PUBLIC -fcolor-diagnostics)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  target_compile_options(tatooine PUBLIC -fdiagnostics-color=always)
endif()
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
#-------------------------------------------------------------------------------
# Subdirectories
#-------------------------------------------------------------------------------
option(TATOOINE_BUILD_DOC "Builds the documentation" OFF)
if (TATOOINE_BUILD_DOC) 
  option(TATOOINE_DOC_ONLY "Only builds documentation" OFF)
  add_subdirectory(doc)
endif () 

if (NOT TATOOINE_DOC_ONLY)
add_subdirectory(gl)
add_subdirectory(test)
add_subdirectory(scivis_contest_2020 EXCLUDE_FROM_ALL)
add_subdirectory(benchmarks)
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
#add_subdirectory(smearing_scalar_field)

if (TATOOINE_MPI_AVAILABLE)
  add_subdirectory(mpi)
  add_subdirectory(insitu)
endif()

option(TATOOINE_BUILD_CHANNELFLOW_VISUALIZATION
       "creates targets to make visualizations for the channelflow" ON)
if(TATOOINE_BUILD_CHANNELFLOW_VISUALIZATION)
  add_subdirectory(channelflow)
endif()


set(TATOOINE_MANDATORY_GIT_SUBMODULES external/blaspp external/lapackpp external/indicators)
foreach(SUBMODULE ${TATOOINE_MANDATORY_GIT_SUBMODULES})
  update_git_submodule(${SUBMODULE})
endforeach()
#===============================================================================
# Required Packages
#===============================================================================
# OpenMP
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
find_package(OpenMP REQUIRED COMPONENTS CXX)
option(TATOOINE_COMPILE_WITH_OPENMP "" ON)
if (TATOOINE_COMPILE_WITH_OPENMP)
  set(TATOOINE_OPENMP_AVAILABLE 1)
  target_link_libraries(tatooine PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(tatooine PUBLIC TATOOINE_OPENMP_AVAILABLE)
  message(STATUS "Compiling with OpenMP")
else()
  set(TATOOINE_OPENMP_AVAILABLE 0)
  message(STATUS "OpenMP not available")
endif()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# BLAS
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
add_subdirectory(external/blaspp)
target_compile_options(blaspp PUBLIC
  -Wno-unused-parameter -Wno-unused-private-field)
target_link_libraries(tatooine PUBLIC blaspp)
list(APPEND TATOOINE_EXPORTING_TARGETS blaspp)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# LAPACK
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
add_subdirectory(external/lapackpp)
target_compile_options(lapackpp PUBLIC -Wno-return-type-c-linkage)
target_compile_definitions(tatooine PUBLIC LAPACK_COMPLEX_CPP)
target_link_libraries(tatooine PUBLIC lapackpp)
list(APPEND TATOOINE_EXPORTING_TARGETS lapackpp)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Boost
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if(TATOOINE_STD_FILESYSTEM_AVAILABLE)
  find_package(Boost REQUIRED COMPONENTS program_options)
  target_link_libraries(tatooine PUBLIC
    ${Boost_PROGRAM_OPTIONS_LIBRARY})
else()
  find_package(Boost REQUIRED COMPONENTS program_options filesystem)
  target_link_libraries(tatooine PUBLIC
    ${Boost_SYSTEM_LIBRARY} ${Boost_FILESYSTEM_LIBRARY})
endif()
target_compile_definitions(tatooine PUBLIC BOOST_BIND_GLOBAL_PLACEHOLDERS)

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
# CDT
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
option(TATOOINE_USE_CDT "By setting this to ON tatooine will checkout CDT from github and compile it." ON)
if (TATOOINE_USE_CDT)
  update_git_submodule(${TATOOINE_EXTERNAL}/CDT)
  set(CDT_USE_AS_COMPILED_LIBRARY ON CACHE BOOL "If enabled templates for float and double will be instantiated and compiled into a library")
  set(CDT_USE_BOOST ON CACHE BOOL "If enabled Boost::rtree can be used for finding the nearest point; boost::flat_set is used for better performance")
  add_subdirectory(${TATOOINE_EXTERNAL}/CDT/CDT)
  target_link_libraries(tatooine PUBLIC CDT)
  target_compile_definitions(tatooine PUBLIC TATOOINE_CDT_AVAILABLE=1)
  list(APPEND TATOOINE_EXPORTING_TARGETS CDT)
else()
  target_compile_definitions(tatooine PUBLIC TATOOINE_CDT_AVAILABLE=0)
endif()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# HDF5
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
option(TATOOINE_USE_SYSTEM_HDF5_LIBS "By setting this to ON tatooine will not compile HDF5 on its own." ON)
if (TATOOINE_USE_SYSTEM_HDF5_LIBS) 
  find_package(HDF5 QUIET COMPONENTS C)
  if(HDF5_FOUND)
    if (HDF5_IS_PARALLEL)
      message(STATUS "HDF5 available with MPI support")
    else()
      message(STATUS "HDF5 available without MPI support")
    endif()
    target_compile_definitions(tatooine PUBLIC
      ${HDF5_DEFINITIONS} ${HDF5_C_DEFINITIONS})
    set(TATOOINE_HDF5_AVAILABLE 1) 
    target_link_libraries(tatooine PUBLIC
      ${HDF5_LIBRARIES})
    target_include_directories(tatooine PUBLIC
      ${HDF5_INCLUDE_DIRS})
  else()
    message(STATUS "HDF5 not available")
  endif()
  set(TATOOINE_HDF5_HEADER hdf5.h)
elseif ((NOT TATOOINE_USE_SYSTEM_HDF5_LIBS) OR
    (TATOOINE_USE_SYSTEM_HDF5_LIBS AND NOT HDF5_FOUND))
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


  target_link_libraries(tatooine PUBLIC hdf5-static)
  set(TATOOINE_HDF5_AVAILABLE 1) 
  list(APPEND TATOOINE_EXPORTING_TARGETS hdf5-static)
  target_compile_definitions(tatooine PUBLIC TATOOINE_USE_SYSTEM_HDF5_LIBS=0)
  set(TATOOINE_HDF5_HEADER ${TATOOINE_EXTERNAL}/hdf5/src/hdf5.h)
else()
  set(TATOOINE_HDF5_AVAILABLE 0) 
endif()
target_compile_definitions(tatooine PUBLIC
  TATOOINE_HDF5_AVAILABLE=${TATOOINE_HDF5_AVAILABLE})
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/hdf5_include.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/include/tatooine/hdf5_include.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/hdf5/type.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/include/tatooine/hdf5/type.h)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# NetCDF
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if(TATOOINE_USE_SYSTEM_HDF5_LIBS)
#  option(TATOOINE_USE_SYSTEM_NETCDF_LIBS
#         "By setting this to ON tatooine will not compile NetCDF on its own."
#         OFF)
#  if (TATOOINE_USE_SYSTEM_NETCDF_LIBS)
    find_package(NetCDF QUIET)
    if (NETCDF_FOUND)
      message(STATUS "NetCDF available")
      set(TATOOINE_NETCDF_AVAILABLE 1)
      target_compile_definitions(tatooine PUBLIC TATOOINE_NETCDF_AVAILABLE=1)
      target_link_libraries(tatooine PUBLIC ${NETCDF_LIBRARIES})
      target_include_directories(tatooine PUBLIC ${NETCDF_INCLUDES})
    else()
      target_compile_definitions(tatooine PUBLIC TATOOINE_NETCDF_AVAILABLE=0)
    endif()
#  endif()
#endif()
#if ((NOT TATOOINE_USE_SYSTEM_NETCDF_LIBS) OR
#  (TATOOINE_USE_SYSTEM_NETCDF_LIBS AND NOT NETCDF_FOUND))
#  set(TATOOINE_NETCDF_AVAILABLE 0)
#  target_compile_definitions(tatooine PUBLIC TATOOINE_NETCDF_AVAILABLE=0)
#  #update_git_submodule(${TATOOINE_EXTERNAL}/netcdf-c)
#  #update_git_submodule(${TATOOINE_EXTERNAL}/netcdf-cxx4)
#  #add_subdirectory(external/netcdf-c EXCLUDE_FROM_ALL)
#  #add_subdirectory(external/netcdf-cxx4 EXCLUDE_FROM_ALL)
endif()

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# FLANN
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
find_package(FLANN QUIET)
if (FLANN_FOUND)
  target_compile_definitions(tatooine PUBLIC TATOOINE_FLANN_AVAILABLE=1)
  target_link_libraries(tatooine PUBLIC FLANN::FLANN)
  message(STATUS "FLANN available")
else()
  target_compile_definitions(tatooine PUBLIC TATOOINE_FLANN_AVAILABLE=0)
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


option(TATOOINE_BUILD_EXAMPLES "")
if (TATOOINE_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
endif()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# This makes the project importable from the build directory
#export(TARGETS ${TATOOINE_EXPORTING_TARGETS}
#       NAMESPACE tatooine::
#       FILE tatooineConfig.cmake
#       EXPORT_LINK_INTERFACE_LIBRARIES)
export(PACKAGE tatooine)
