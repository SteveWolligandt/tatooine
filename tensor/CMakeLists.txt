#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
add_library(tensor INTERFACE)
set_property(TARGET tensor PROPERTY CXX_STANDARD 20)
target_include_directories(tensor INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
target_link_libraries(tensor
 INTERFACE
  base
  multidim_array
)

find_package(BLAS)
find_package(LAPACK)
if (BLAS_FOUND AND LAPACK_FOUND)
  option(TATOOINE_USE_BLAS_AND_LAPACK "Use BLAS and LAPACK" ON)
  set(TATOOINE_USE_BLAS_AND_LAPACK ${TATOOINE_USE_BLAS_AND_LAPACK} PARENT_SCOPE)
  if(TATOOINE_USE_BLAS_AND_LAPACK)
    target_link_libraries(tensor INTERFACE BLAS::BLAS LAPACK::LAPACK)
    #target_compile_options(lapackpp INTERFACE -Wno-unused-parameter)
    target_compile_definitions(tensor INTERFACE LAPACK_COMPLEX_CPP)
    set(TATOOINE_BLAS_AND_LAPACK_AVAILABLE 1)
  else()
    set(TATOOINE_BLAS_AND_LAPACK_AVAILABLE 0)
  endif()
else()
    set(TATOOINE_BLAS_AND_LAPACK_AVAILABLE 0)
endif()
target_compile_definitions(tensor INTERFACE
TATOOINE_BLAS_AND_LAPACK_AVAILABLE=${TATOOINE_BLAS_AND_LAPACK_AVAILABLE})
set(TATOOINE_BLAS_AND_LAPACK_AVAILABLE ${TATOOINE_BLAS_AND_LAPACK_AVAILABLE} PARENT_SCOPE)
