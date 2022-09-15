enable_language(Fortran)
add_executable(mpi.fortran.hello_world
  hello_world.f90
  c_interface.f90)
add_custom_target(mpi.fortran.hello_world.run
  ./mpi.fortran.hello_world
  DEPENDS mpi.fortran.hello_world)

add_library(mpi.fortran.c_interface SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/c_interface.cpp)
set_property(TARGET mpi.fortran.c_interface PROPERTY CXX_STANDARD 20)

target_link_libraries(mpi.fortran.hello_world mpi.fortran.c_interface)
