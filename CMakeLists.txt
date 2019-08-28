cmake_minimum_required(VERSION 3.10)
project(tatooine)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")

find_package(GiNaC REQUIRED)

# create library target
add_library(tatooine INTERFACE)
target_compile_features(tatooine INTERFACE cxx_std_17)

# set include dirs
target_include_directories(tatooine
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GINAC_INCLUDE_DIRS}
)

#set libs 
target_link_libraries(tatooine
  INTERFACE
    ${GINAC_LIBRARIES}
)

add_subdirectory(test)
