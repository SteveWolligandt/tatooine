list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/")
add_library(preprocessor INTERFACE)
target_include_directories(preprocessor INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/include)

set(TATOOINE_PP_NUM_MAX_INDICES "100" CACHE STRING "")
include(preprocessor_setup)
