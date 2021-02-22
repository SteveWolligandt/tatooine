find_package(MPI COMPONENTS C CXX)
if (MPI_C_FOUND)
  #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  # test interface
  #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  add_library(tatooine.mpi SHARED src/program.cpp)
  find_package(Boost COMPONENTS mpi REQUIRED)
  target_compile_options(tatooine.mpi PUBLIC ${MPI_C_COMPILE_OPTIONS} ${MPI_CXX_COMPILE_OPTIONS})
  target_compile_definitions(tatooine.mpi PUBLIC ${MPI_C_COMPILE_DEFINITIONS} ${MPI_CXX_COMPILE_DEFINITIONS})
  target_link_libraries(tatooine.mpi PUBLIC tatooine ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES} Boost::mpi)
  target_include_directories(tatooine.mpi PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include ${MPI_C_INCLUDE_DIRS} ${MPI_CXX_INCLUDE_DIRS})
  #target_link_options(tatooine.mpi PUBLIC ${MPI_C_LINK_FLAGS} ${MPI_CXX_LINK_FLAGS})
  list(APPEND TATOOINE_EXPORTING_TARGETS tatooine.mpi)
endif()
