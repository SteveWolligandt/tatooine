add_library(fields INTERFACE)
target_include_directories(fields INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(fields INTERFACE tensor geometry)
