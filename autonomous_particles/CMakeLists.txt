if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "-fcolor-diagnostics")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "-fdiagnostics-color=always")
endif()
#===============================================================================
function(create_autonomous_particles_application CPP_FILE)
  set(TAT_AP_APP autonomous_particles.${CPP_FILE})
  add_executable (${TAT_AP_APP} ${CPP_FILE}.cpp parse_args.cpp)
  target_link_libraries(${TAT_AP_APP} tatooine Boost::program_options)
  target_compile_options(${TAT_AP_APP} PRIVATE
  -Wall -Wextra -pedantic -Wno-missing-braces -Wno-unused-lambda-capture
  -march=native)
add_custom_target(${TAT_AP_APP}.run
  COMMAND ./autonomous_particles.${CPP_FILE}
  DEPENDS ${TAT_AP_APP})

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  add_custom_command(OUTPUT ${TAT_AP_APP}.callgrind
    COMMAND valgrind --tool=callgrind --callgrind-out-file=${TAT_AP_APP}.callgrind ./${TAT_AP_APP}
    DEPENDS ${TAT_AP_APP})
  add_custom_target(${TAT_AP_APP}.callgrind.show
    COMMAND kcachegrind ${TAT_AP_APP}.callgrind
    DEPENDS ${TAT_AP_APP}.callgrind)

  add_custom_command(OUTPUT ${TAT_AP_APP}.callgrind.dot
    COMMAND gprof2dot --format=callgrind --output=${TAT_AP_APP}.dot ${TAT_AP_APP}.callgrind
    DEPENDS ${TAT_AP_APP}.callgrind)
  add_custom_target(${TAT_AP_APP}.callgrind.dot.show
    COMMAND dot -Tsvg ${TAT_AP_APP}.dot -o ${TAT_AP_APP}-graph.svg
    DEPENDS ${TAT_AP_APP}.callgrind.dot)
endif()
unset(TAT_AP_APP)
endfunction()
create_autonomous_particles_application(doublegyre_grid)
create_autonomous_particles_application(abcflow_grid)
