add_executable(channelflow.cutting_planes cutting_planes.cpp)
target_link_libraries(channelflow.cutting_planes PUBLIC tatooine)
add_custom_target(
  channelflow.cutting_planes.run
  COMMAND ./channelflow.cutting_planes
  DEPENDS channelflow.cutting_planes)
# force coloring
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  target_compile_options(channelflow.cutting_planes PRIVATE "-fcolor-diagnostics")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  target_compile_options(channelflow.cutting_planes PRIVATE  "-fdiagnostics-color=always")
endif()
#===============================================================================
add_executable(channelflow.data_check data_check.cpp)
target_link_libraries(channelflow.data_check PUBLIC tatooine)
add_custom_target(
  channelflow.data_check.run
  COMMAND ./channelflow.data_check
  DEPENDS channelflow.data_check)
# force coloring
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  target_compile_options(channelflow.data_check PRIVATE "-fcolor-diagnostics")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  target_compile_options(channelflow.data_check PRIVATE  "-fdiagnostics-color=always")
endif()
#===============================================================================
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
add_executable(channelflow_visualization.direct_volume direct_volume.cpp)
target_link_libraries(channelflow_visualization.direct_volume PUBLIC tatooine)
add_custom_target(
  channelflow_visualization.direct_volume.run
  COMMAND ./channelflow_visualization.direct_volume
  DEPENDS channelflow_visualization.direct_volume)
# force coloring
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  target_compile_options(channelflow_visualization.direct_volume PRIVATE "-fcolor-diagnostics")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  target_compile_options(channelflow_visualization.direct_volume PRIVATE  "-fdiagnostics-color=always")
endif()
#===============================================================================
add_executable(channelflow_visualization.direct_isosurface direct_isosurface.cpp)
target_link_libraries(channelflow_visualization.direct_isosurface PUBLIC tatooine)
add_custom_target(
  channelflow_visualization.direct_isosurface.run
  COMMAND ./channelflow_visualization.direct_isosurface
  DEPENDS channelflow_visualization.direct_isosurface)
# force coloring
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  target_compile_options(channelflow_visualization.direct_isosurface
                         PRIVATE "-fcolor-diagnostics")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  target_compile_options(channelflow_visualization.direct_isosurface
                         PRIVATE  "-fdiagnostics-color=always")
endif()
#===============================================================================
add_executable(channelflow_visualization.add_velocity_magnitude create_velocity_magnitude.cpp)
target_link_libraries(channelflow_visualization.add_velocity_magnitude PUBLIC tatooine)
add_custom_target(
  channelflow_visualization.add_velocity_magnitude.run
  COMMAND ./channelflow_visualization.add_velocity_magnitude
  DEPENDS channelflow_visualization.add_velocity_magnitude)
# force coloring
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  target_compile_options(channelflow_visualization.add_velocity_magnitude PRIVATE "-fcolor-diagnostics")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  target_compile_options(channelflow_visualization.add_velocity_magnitude PRIVATE  "-fdiagnostics-color=always")
endif()
#===============================================================================
add_executable(channelflow_visualization.add_Q_pnorm create_Q.cpp)
target_link_libraries(channelflow_visualization.add_Q_pnorm PUBLIC tatooine)
add_custom_target(
  channelflow_visualization.add_Q_pnorm.run
  COMMAND ./channelflow_visualization.add_Q_pnorm
  DEPENDS channelflow_visualization.add_Q_pnorm)
# force coloring
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  target_compile_options(channelflow_visualization.add_Q_pnorm PRIVATE "-fcolor-diagnostics")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  target_compile_options(channelflow_visualization.add_Q_pnorm PRIVATE  "-fdiagnostics-color=always")
endif()
#===============================================================================
if (TATOOINE_MPI_AVAILABLE)
  add_executable(channelflow.pv.mpi pv_mpi.cpp)

  target_link_libraries(channelflow.pv.mpi PUBLIC tatooine tatooine.mpi)
  add_custom_target(channelflow.pv.mpi.run
    mpirun -np 8 --host localhost:8 ./channelflow.pv.mpi
    DEPENDS tatooine channelflow.pv.mpi)
  # force coloring
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_compile_options(channelflow.pv.mpi PRIVATE "-fcolor-diagnostics")
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(channelflow.pv.mpi PRIVATE  "-fdiagnostics-color=always")
  endif()
endif()
