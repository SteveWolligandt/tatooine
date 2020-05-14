cmake_minimum_required(VERSION 3.15)
project(tatooine)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules/")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/external/catch2/contrib/")

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
add_subdirectory(external/catch2)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL
    "Enable building the unit tests which depend on gtest")
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL
    "Enable testing of the benchmark library.")
add_subdirectory(external/google-benchmark)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
find_package(TBB REQUIRED)
find_package(yavin REQUIRED)
find_package(GiNaC QUIET)
find_package(LAPACKE REQUIRED)
find_package(OpenMP REQUIRED)
find_package(vcode REQUIRED)
find_package(PNG REQUIRED)
find_package(PNG++ REQUIRED)
find_package(HDF5 REQUIRED)
set(HDF5_LIBS -lhdf5 -lhdf5_cpp -lhdf5_hl)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# configure files
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
set(TATOOINE_LIC_COMP_SHADER
    ${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/gpu/lic.comp)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/gpu/lic.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/include/tatooine/gpu/lic.h")

set(TATOOINE_LINE_SHADER_VERT_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/gpu/line.vert)
set(TATOOINE_LINE_SHADER_GEOM_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/gpu/line.geom)
set(TATOOINE_LINE_SHADER_FRAG_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/gpu/line.frag)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/gpu/line_shader.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/include/tatooine/gpu/line_shader.h")

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
add_library(tatooine INTERFACE)
target_compile_features(tatooine INTERFACE cxx_std_17)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# set include dirs
target_include_directories(tatooine
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/single_include
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${LAPACKE_INCLUDE_DIR}
    ${vcode_DIR}/../..
    ${PNG_INCLUDE_DIRS}
    ${PNG++_INCLUDE_DIRS}
)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#set libs 
target_link_libraries(tatooine
  INTERFACE
    yavin
    tbb
    ${LAPACKE_LIBRARIES}
    ${PNG_LIBRARIES}
    ${PNG++_LIBRARIES}
    ${HDF5_LIBS}
)

if (GiNaC_FOUND) 
  set(TATOOINE_GINAC_AVAILABLE 1)
  target_include_directories(tatooine INTERFACE ${GINAC_INCLUDE_DIRS})
  target_link_libraries(tatooine INTERFACE ${GINAC_LIBRARIES})
else()
  set(TATOOINE_GINAC_AVAILABLE 0)
endif()

if(OpenMP_CXX_FOUND)
  set(TATOOINE_OPENMP_AVAILABLE 1)
  target_link_libraries(tatooine INTERFACE OpenMP::OpenMP_CXX)
else()
  set(TATOOINE_OPENMP_AVAILABLE 0)
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/packages.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/include/tatooine/packages.h)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
add_subdirectory(test)
add_subdirectory(benchmarks)
add_subdirectory(doc)
#add_subdirectory(misc)
add_subdirectory(steadification)
add_subdirectory(material_science)
add_subdirectory(flowexplorer)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# This makes the project importable from the build directory
export(TARGETS tatooine FILE tatooineConfig.cmake)
export(PACKAGE tatooine)
