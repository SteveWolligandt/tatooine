#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
add_library(tensor INTERFACE)
target_compile_features(tensor INTERFACE cxx_std_20)
target_include_directories(tensor INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
update_git_submodule(${CMAKE_CURRENT_SOURCE_DIR}/external/blaspp)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/blaspp)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
update_git_submodule(${CMAKE_CURRENT_SOURCE_DIR}/external/lapackpp)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/lapackpp)
target_compile_definitions(tensor INTERFACE LAPACK_COMPLEX_CPP)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
target_link_libraries(tensor
 INTERFACE
  base
  multidim_array
  blaspp
  lapackpp)
