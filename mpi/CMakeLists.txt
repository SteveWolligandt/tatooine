set(TATOMPI_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(TATOMPI_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(TATOMPI_BINARY_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/src)
set(TATOMPI_BINARY_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
set(TATOMPI_INTERFACE_SOURCE_DIR ${TATOMPI_SOURCE_DIR}/interfaces)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
find_package(MPI)
if (MPI_FOUND)
  FIND_PACKAGE(Boost COMPONENTS mpi REQUIRED)
  #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  # test interface
  #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  add_library(mpi.interface.test STATIC
              ${TATOMPI_INTERFACE_SOURCE_DIR}/test.cpp)
  target_link_libraries(mpi.interface.test tatooine ${MPI_LIBRARIES} Boost::mpi)
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_compile_options(mpi.interface.test
      PRIVATE "-fcolor-diagnostics")
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(mpi.interface.test
      PRIVATE  "-fdiagnostics-color=always")
  endif()
  if (MPI_COMPILE_FLAGS)
    set_target_properties(mpi.interface.test PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
  endif()
  if(MPI_LINK_FLAGS)
    set_target_properties(mpi.interface.test PROPERTIES
      LINK_FLAGS "${MPI_LINK_FLAGS}")
  endif()
  target_compile_features(mpi.interface.test PUBLIC cxx_std_20)
  target_compile_options(mpi.interface.test
    PRIVATE
    -Wall -Wextra -pedantic
    -Wno-tautological-undefined-compare
    -Wno-gnu-zero-variadic-macro-arguments)
  target_include_directories(mpi.interface.test
    PUBLIC ${TATOMPI_INCLUDE_DIR} ${TATOMPI_BINARY_INCLUDE_DIR})

  #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  # feeder executable
  #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  add_executable(mpi.feeder.test ${TATOMPI_SOURCE_DIR}/feeders/test.cpp
                                 ${TATOMPI_SOURCE_DIR}/feeders/analytical_function.cpp)
  target_link_libraries(mpi.feeder.test
    tatooine ${MPI_LIBRARIES} Boost::program_options mpi.interface.test)
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_compile_options(mpi.feeder.test PRIVATE "-fcolor-diagnostics")
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(mpi.feeder.test PRIVATE  "-fdiagnostics-color=always")
  endif()

  if (MPI_COMPILE_FLAGS)
    set_target_properties(mpi.feeder.test PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
  endif()
  if(MPI_LINK_FLAGS)
    set_target_properties(mpi.feeder.test PROPERTIES
      LINK_FLAGS "${MPI_LINK_FLAGS}")
  endif()
  target_compile_features(mpi.feeder.test PUBLIC cxx_std_20)
  target_compile_options(mpi.feeder.test
    PRIVATE
    -Wall -Wextra -pedantic
    -Wno-tautological-undefined-compare -Wno-gnu-zero-variadic-macro-arguments)
  target_include_directories(mpi.feeder.test
    PUBLIC ${TATOMPI_INCLUDE_DIR} ${TATOMPI_BINARY_INCLUDE_DIR})
  add_custom_target(mpi.feeder.test.run
    mpirun -np 4 ./mpi.feeder.test 
    --gridsizex               20 
    --gridsizey               20 
    --gridsizez               20 
    --max_init_distance       0.1 
    --normal_error_max        0.1 
    --normal_error_min        0.01 
    --saving_tracers_interval 1
    --tangential_error_max    0.1 
    --tangential_error_min    0.01
    --t0                      0
    --t1                      10
    --dt                      1
    DEPENDS tatooine mpi.feeder.test)
else()
  message("MPI not found.")
endif()

add_subdirectory(fortran)
