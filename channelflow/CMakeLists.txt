add_executable(channelflow_visualization channelflow_visualization.cpp)
target_link_libraries(channelflow_visualization PUBLIC tatooine)

add_custom_target(
  channelflow_visualization.run
  COMMAND ./channelflow_visualization
  DEPENDS channelflow_visualization)
# force coloring
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  target_compile_options(channelflow_visualization PRIVATE "-fcolor-diagnostics")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  target_compile_options(channelflow_visualization PRIVATE  "-fdiagnostics-color=always")
endif()
#===============================================================================
option(TATOOINE_COMPILE_FOR_SUPERMUC "This adds the psm2 compatibility library to the linker" OFF)
if (TATOOINE_COMPILE_FOR_SUPERMUC)
  link_directories(/usr/lib64/psm2-compat)
endif()

add_executable(channelflow.pv.mpi pv_mpi.cpp)

if (TATOOINE_COMPILE_FOR_SUPERMUC)
  target_link_libraries(channelflow.pv.mpi PUBLIC /usr/lib64/psm2-compat/libpsm_infinipath.so.1)
endif()
target_link_libraries(channelflow.pv.mpi PUBLIC tatooine mpi)
add_custom_target(channelflow.pv.mpi.run
  mpirun -np 8 --host localhost:8 ./channelflow.pv.mpi
  DEPENDS tatooine channelflow.pv.mpi)
# force coloring
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  target_compile_options(channelflow.pv.mpi PRIVATE "-fcolor-diagnostics")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  target_compile_options(channelflow.pv.mpi PRIVATE  "-fdiagnostics-color=always")
endif()
