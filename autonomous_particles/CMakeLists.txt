if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "-fcolor-diagnostics")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "-fdiagnostics-color=always")
endif()
#===============================================================================
set(TAT_AP_DG_GRID autonomous_particles.doublegyre_grid)
add_executable (${TAT_AP_DG_GRID} doublegyre_grid.cpp)
target_link_libraries(${TAT_AP_DG_GRID} tatooine Boost::program_options)
target_compile_features(${TAT_AP_DG_GRID} PUBLIC cxx_std_20)
target_compile_options(${TAT_AP_DG_GRID} PRIVATE
  -Wall -Wextra -pedantic -Wno-missing-braces -Wno-unused-lambda-capture
  -march=native)

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
add_custom_command(
  OUTPUT ${TAT_AP_DG_GRID}.callgrind
  COMMAND valgrind --tool=callgrind --callgrind-out-file=${TAT_AP_DG_GRID}.callgrind ./${TAT_AP_DG_GRID}
  DEPENDS ${TAT_AP_DG_GRID})
add_custom_target(
  ${TAT_AP_DG_GRID}.callgrind.show
  COMMAND kcachegrind ${TAT_AP_DG_GRID}.callgrind
  DEPENDS ${TAT_AP_DG_GRID}.callgrind)

add_custom_command(
  OUTPUT ${TAT_AP_DG_GRID}.callgrind.dot
  COMMAND gprof2dot --format=callgrind --output=${TAT_AP_DG_GRID}.dot ${TAT_AP_DG_GRID}.callgrind
  DEPENDS ${TAT_AP_DG_GRID}.callgrind)
add_custom_target(
  ${TAT_AP_DG_GRID}.callgrind.dot.show
  COMMAND dot -Tsvg ${TAT_AP_DG_GRID}.dot -o ${TAT_AP_DG_GRID}-graph.svg
  DEPENDS ${TAT_AP_DG_GRID}.callgrind.dot)
endif()
