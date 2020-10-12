if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "-fcolor-diagnostics")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "-fdiagnostics-color=always")
endif()
#===============================================================================
add_executable (autonomous_particles.dg_grid dg_grid.cpp)
target_link_libraries(autonomous_particles.dg_grid tatooine Boost::program_options)
target_compile_features(autonomous_particles.dg_grid PUBLIC cxx_std_20)
target_compile_options(autonomous_particles.dg_grid PRIVATE
  -Wall -Wextra -pedantic -Wno-missing-braces -Wno-unused-lambda-capture
  -march=native)
