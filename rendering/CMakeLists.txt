add_library(rendering INTERFACE)
target_include_directories(rendering INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  #${CMAKE_CURRENT_BINARY_DIR}/include
  )
target_link_libraries(rendering INTERFACE gl geometry)
