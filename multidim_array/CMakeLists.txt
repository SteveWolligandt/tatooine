add_library(multidim_array INTERFACE)
target_include_directories(multidim_array
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(multidim_array INTERFACE base)
