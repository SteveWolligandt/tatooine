set(TATOOINE_DINO_INTERFACE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(TATOOINE_DINO_INTERFACE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(TATOOINE_DINO_INTERFACE_BINARY_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/src)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
find_package(MPI)
if (MPI_FOUND)
  find_package(Boost COMPONENTS mpi REQUIRED)
  #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  # test interface
  #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  add_library(dino_interface STATIC
    ${TATOOINE_DINO_INTERFACE_SOURCE_DIR}/interface.cpp)
  target_link_libraries(dino_interface PRIVATE tatooine ${MPI_LIBRARIES} Boost::mpi)
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_compile_options(dino_interface
      PRIVATE "-fcolor-diagnostics")
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(dino_interface
      PRIVATE  "-fdiagnostics-color=always")
  endif()
  if (MPI_COMPILE_FLAGS)
    set_target_properties(dino_interface PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
  endif()
  if(MPI_LINK_FLAGS)
    set_target_properties(dino_interface PROPERTIES
      LINK_FLAGS "${MPI_LINK_FLAGS}")
  endif()
  target_compile_features(dino_interface PUBLIC cxx_std_20)
  target_compile_options(dino_interface
    PRIVATE
    -Wall -Wextra -pedantic)
  target_include_directories(dino_interface
    PUBLIC ${TATOOINE_DINO_INTERFACE_INCLUDE_DIR})

  #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  # feeder executable
  #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  add_executable(dino_interface.feeder
    ${TATOOINE_DINO_INTERFACE_SOURCE_DIR}/feeder.cpp)
  target_link_libraries(dino_interface.feeder PRIVATE
    tatooine ${MPI_LIBRARIES} Boost::program_options dino_interface)
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_compile_options(dino_interface.feeder PRIVATE "-fcolor-diagnostics")
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(dino_interface.feeder PRIVATE
      "-fdiagnostics-color=always")
  endif()

  if (MPI_COMPILE_FLAGS)
    set_target_properties(dino_interface.feeder PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
  endif()
  if(MPI_LINK_FLAGS)
    set_target_properties(dino_interface.feeder PROPERTIES
      LINK_FLAGS "${MPI_LINK_FLAGS}")
  endif()
  target_compile_features(dino_interface.feeder PUBLIC cxx_std_20)
  target_compile_options(dino_interface.feeder
    PRIVATE
    -Wall -Wextra -pedantic
    -Wno-tautological-undefined-compare -Wno-gnu-zero-variadic-macro-arguments)
  target_include_directories(dino_interface.feeder
    PUBLIC ${TATOOINE_DINO_INTERFACE_INCLUDE_DIR})
  add_custom_target(dino_interface.feeder.run
    mpirun -np 4 ./dino_interface.feeder 
    --gridsizex 101 
    --gridsizey 101 
    --gridsizez 101 
    --x0         -10
    --x1         10
    --y0         -10
    --y1         10
    --z0         -10
    --z1         10
    --t0         0
    --t1        10
    --dt         1
    DEPENDS tatooine dino_interface.feeder)
else()
  message("MPI not found.")
endif()

add_subdirectory(fortran)
