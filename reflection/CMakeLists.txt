add_library(reflection INTERFACE)
target_include_directories(reflection INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(reflection INTERFACE base preprocessor)
