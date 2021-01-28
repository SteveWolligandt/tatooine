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
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  # Update submodules as needed
  execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  RESULT_VARIABLE GIT_SUBMOD_RESULT)
  if(NOT GIT_SUBMOD_RESULT EQUAL "0")
    message(FATAL_ERROR
      "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
  endif()
endif()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL
    "Enable building the unit tests which depend on gtest")
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL
    "Enable testing of the benchmark library.")
add_subdirectory(external/google-benchmark)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(LAPACKE REQUIRED)
option(TATOOINE_INCLUDE_MKL_LAPACKE "If ON mkl_lapacke.h will be include instead of lapacke.h")
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yavin QUIET)
find_package(vcode REQUIRED)
find_package(PNG QUIET)
if(${PNG_FOUND})
  find_package(PNG++ QUIET)
  if (NOT DEFINED PNG++_INCLUDE_DIRS)
    set(TATOOINE_HAS_PNG_SUPPORT ON)
  else()
    set(TATOOINE_HAS_PNG_SUPPORT OFF)
  endif()
else()
  set(TATOOINE_HAS_PNG_SUPPORT OFF)
endif()
find_package(NetCDF QUIET)
if (${NETCDF_FOUND})
  if (NOT DEFINED PNG++_INCLUDE_DIRS)
    set(TATOOINE_HAS_NETCDF_SUPPORT ON)
  else()
    set(TATOOINE_HAS_NETCDF_SUPPORT OFF)
  endif()
endif()


option (TATOOINE_COMPILE_WITH_OPENMP "" ON)
if (TATOOINE_COMPILE_WITH_OPENMP)
find_package(OpenMP QUIET)
else()
set(TATOOINE_OPENMP_AVAILABLE 0)
endif()

find_package(FLANN QUIET)
if (FLANN_FOUND)
  set(TATOOINE_HAS_FLANN_SUPPORT ON)
else()
  set(TATOOINE_HAS_FLANN_SUPPORT OFF)
endif()

#find_package(TBB REQUIRED)

set(CGAL_DO_NOT_WARN_ABOUT_CMAKE_BUILD_TYPE TRUE)
find_package(CGAL QUIET)
if (CGAL_FOUND)
  set(TATOOINE_HAS_CGAL_SUPPORT ON)
else()
  set(TATOOINE_HAS_CGAL_SUPPORT OFF)
endif()

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
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  target_compile_options(tatooine PRIVATE "-fcolor-diagnostics")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  target_compile_options(tatooine PRIVATE  "-fdiagnostics-color=always")
endif()
if (TATOOINE_INCLUDE_MKL_LAPACKE)
  target_compile_definitions(tatooine PUBLIC TATOOINE_INCLUDE_MKL_LAPACKE)
endif()
if (TATOOINE_HAS_FLANN_SUPPORT)
  target_compile_definitions(tatooine PUBLIC TATOOINE_HAS_FLANN_SUPPORT)
endif()
if (TATOOINE_HAS_PNG_SUPPORT)
  target_compile_definitions(tatooine PUBLIC TATOOINE_HAS_PNG_SUPPORT)
endif()
if (TATOOINE_HAS_NETCDF_SUPPORT)
  target_compile_definitions(tatooine PUBLIC TATOOINE_HAS_NETCDF_SUPPORT)
endif()
if (TATOOINE_HAS_CGAL_SUPPORT)
  target_compile_definitions(tatooine PUBLIC TATOOINE_HAS_CGAL_SUPPORT)
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR
    "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
target_compile_options(tatooine PRIVATE
  -Wall -Wextra -pedantic
  -Wno-tautological-undefined-compare -Wno-gnu-zero-variadic-macro-arguments)
endif()
target_compile_features(tatooine PUBLIC cxx_std_20)
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
)
if(${TATOOINE_HAS_PNG_SUPPORT})
target_include_directories(tatooine PUBLIC ${PNG_INCLUDE_DIRS} ${PNG++_INCLUDE_DIRS})
endif()
if(${TATOOINE_HAS_NETCDF_SUPPORT})
target_include_directories(tatooine PUBLIC ${NETCDF_INCLUDES})
endif()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#set libs 
target_link_libraries(tatooine
  PUBLIC
    stdc++fs
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)
if(${TATOOINE_HAS_FLANN_SUPPORT})
target_link_libraries(tatooine PUBLIC FLANN::FLANN)
endif()
if(${TATOOINE_HAS_PNG_SUPPORT})
target_link_libraries(tatooine PUBLIC ${PNG_LIBRARIES} ${PNG++_LIBRARIES})
endif()
if(${TATOOINE_HAS_NETCDF_SUPPORT})
target_link_libraries(tatooine PUBLIC ${NETCDF_LIBRARIES})
endif()
if(${TATOOINE_HAS_CGAL_SUPPORT})
target_link_libraries(tatooine PUBLIC CGAL::CGAL)
endif()
if (${yavin_FOUND}) 
  set(TATOOINE_YAVIN_AVAILABLE 1)
  target_link_libraries(tatooine PUBLIC yavin)
else()
  set(TATOOINE_YAVIN_AVAILABLE 0)
endif()
#-------------------------------------------------------------------------------
# GINAC
#-------------------------------------------------------------------------------
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

if (TATOOINE_COMPILE_WITH_OPENMP)
if(OpenMP_CXX_FOUND)
  set(TATOOINE_OPENMP_AVAILABLE 1)
  target_link_libraries(tatooine PUBLIC OpenMP::OpenMP_CXX)
else()
  set(TATOOINE_OPENMP_AVAILABLE 0)
endif()
endif()


configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/packages.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/include/tatooine/packages.h)

#-------------------------------------------------------------------------------
# Subdirectories
#-------------------------------------------------------------------------------
add_subdirectory(test)
add_subdirectory(scivis_contest_2020)
add_subdirectory(benchmarks)
add_subdirectory(doc)
add_subdirectory(material_science)
if (${yavin_FOUND}) 
  add_subdirectory(steadification)
  add_subdirectory(flowexplorer)
endif()
add_subdirectory(autonomous_particles)
add_subdirectory(smearing_scalar_field)
add_subdirectory(dino_interface)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# This makes the project importable from the build directory
export(TARGETS tatooine dino_interface
       FILE tatooineConfig.cmake
       NAMESPACE tatooine::)
export(PACKAGE tatooine)
