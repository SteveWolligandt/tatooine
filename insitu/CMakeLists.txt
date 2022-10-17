set(TATOOINE_INSITU_INTERFACE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(TATOOINE_INSITU_INTERFACE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(TATOOINE_INSITU_INTERFACE_BINARY_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/src)
# -----------------------------------------------------------------------------
if(TARGET mpi)
  # ---------------------------------------------------------------------------
  # test interface
  # ---------------------------------------------------------------------------
  add_library(
    insitu_interface STATIC
    ${TATOOINE_INSITU_INTERFACE_SOURCE_DIR}/interface.cpp
    ${TATOOINE_INSITU_INTERFACE_SOURCE_DIR}/c_interface.cpp)
  target_link_libraries(insitu_interface PUBLIC mpi)
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_compile_options(insitu_interface PRIVATE "-fcolor-diagnostics")
  elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(insitu_interface
                           PRIVATE "-fdiagnostics-color=always")
  endif()
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}"
                                                  STREQUAL "Clang")
    target_compile_options(insitu_interface PRIVATE -Wall -Wextra -pedantic)
  endif()
  target_include_directories(insitu_interface
                             PUBLIC ${TATOOINE_INSITU_INTERFACE_INCLUDE_DIR})
  list(APPEND TATOOINE_EXPORTING_TARGETS insitu_interface)

  # ---------------------------------------------------------------------------
  # feeder executable
  # ---------------------------------------------------------------------------
  add_executable(insitu_interface.feeder
                 ${TATOOINE_INSITU_INTERFACE_SOURCE_DIR}/feeder.cpp)
  target_link_libraries(insitu_interface.feeder PRIVATE tatooine
                                                        insitu_interface)
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_compile_options(insitu_interface.feeder
                           PRIVATE "-fcolor-diagnostics")
  elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(insitu_interface.feeder
                           PRIVATE "-fdiagnostics-color=always")
  endif()

  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}"
                                                  STREQUAL "Clang")
    target_compile_options(
      insitu_interface.feeder
      PRIVATE -Wall -Wextra -Wpedantic -Wno-tautological-undefined-compare
              -Wno-gnu-zero-variadic-macro-arguments)
  endif()
  target_include_directories(insitu_interface.feeder
                             PUBLIC ${TATOOINE_INSITU_INTERFACE_INCLUDE_DIR})
  add_custom_target(
    insitu_interface.feeder.run
    # mpirun -np 4 --host localhost:4 ./insitu_interface.feeder
    mpirun
    -np
    8
    --host
    localhost:8
    ./insitu_interface.feeder
    --gridsizex
    101
    --gridsizey
    101
    --gridsizez
    101
    --x0
    -10
    --x1
    10
    --y0
    -10
    --y1
    10
    --z0
    -10
    --z1
    10
    --t0
    0
    --t1
    10
    --dt
    0.1
    DEPENDS tatooine insitu_interface.feeder)
endif()

add_subdirectory(fortran)
