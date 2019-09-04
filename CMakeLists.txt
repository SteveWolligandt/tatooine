cmake_minimum_required(VERSION 3.10)
project(tatooine)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
find_package(GiNaC REQUIRED)
find_package(LAPACKE REQUIRED)
find_package(OpenMP REQUIRED)
find_package(vcode REQUIRED)

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
    ${LAPACK_INCLUDE_DIR}
    ${vcode_DIR}/../..
)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#set libs 
target_link_libraries(tatooine
  INTERFACE
    ${GINAC_LIBRARIES}
    ${LAPACK_LIBRARIES}
)
if(OpenMP_CXX_FOUND)
  target_link_libraries(tatooine INTERFACE OpenMP::OpenMP_CXX)
endif()

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
add_subdirectory(test)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# This makes the project importable from the build directory
export(TARGETS tatooine FILE tatooineConfig.cmake)
