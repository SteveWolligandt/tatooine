update_git_submodule(external/blaspp)
update_git_submodule(external/lapackpp)
add_library(tensor INTERFACE)
target_compile_features(tensor INTERFACE cxx_std_20)
target_include_directories(tensor INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
add_subdirectory(external/blaspp)
target_compile_options(blaspp PUBLIC -Wno-unused-parameter)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
add_subdirectory(external/lapackpp)
target_compile_options(lapackpp PUBLIC -Wno-return-type-c-linkage)
target_compile_definitions(tensor INTERFACE LAPACK_COMPLEX_CPP)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
target_link_libraries(tensor INTERFACE
  base
  multidim_array
  blaspp
  lapackpp)
