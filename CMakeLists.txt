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
find_package(yavin REQUIRED)
find_package(GiNaC REQUIRED)
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

set(TATOOINE_LINE_SHADER_VERT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/gpu/line.vert)
set(TATOOINE_LINE_SHADER_GEOM_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/gpu/line.geom)
set(TATOOINE_LINE_SHADER_FRAG_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/gpu/line.frag)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/gpu/line_shader.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/include/tatooine/gpu/line_shader.h")
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
    yavin
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
add_subdirectory(benchmarks)
add_subdirectory(doc)
add_subdirectory(misc)
add_subdirectory(steadification)
add_subdirectory(material_science)
add_subdirectory(flowexplorer)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# This makes the project importable from the build directory
export(TARGETS tatooine FILE tatooineConfig.cmake)
export(PACKAGE tatooine)
