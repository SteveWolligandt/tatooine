find_package(
  Boost
  COMPONENTS program_options
  REQUIRED)
file(GLOB EXAMPLE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
foreach(EXAMPLE_FILE ${EXAMPLE_FILES})
  get_filename_component(FILENAME ${EXAMPLE_FILE} NAME_WE)
  set(TARGET ${FILENAME}.example)
  add_executable(${TARGET} ${FILENAME}.cpp)
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_compile_options(${TARGET} PRIVATE -fcolor-diagnostics)
  elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(${TARGET} PRIVATE -fdiagnostics-color=always)
  endif()
  set_property(TARGET ${TARGET} PROPERTY CXX_STANDARD 20)
  target_link_libraries(${TARGET} PUBLIC fields rendering
                                         Boost::program_options)
  add_custom_target(
    ${TARGET}.run
    COMMAND ./${TARGET}
    DEPENDS ${TARGET})
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # add valgrind targets for unittest
    add_custom_target(
      ${TARGET}.valgrind
      COMMAND valgrind --log-file="${TARGET}.valgrind" --tool=memcheck
              --leak-check=yes --show-reachable=yes --track-fds=yes ./${TARGET}
      DEPENDS ${TARGET})
    add_custom_target(
      ${TARGET}.callgrind
      COMMAND valgrind --tool=callgrind --callgrind-out-file=${TARGET}.callgrind
              --dump-instr=yes --collect-jumps=yes ./${TARGET}
      DEPENDS ${TARGET})
    add_custom_target(
      ${TARGET}.callgrind.kcachegrind
      COMMAND kcachegrind ${TARGET}.callgrind
      DEPENDS ${TARGET}.callgrind)
    add_custom_target(
      ${TARGET}.callgrind.dot
      COMMAND gprof2dot --format=callgrind --output=${TARGET}.dot
              ${TARGET}.callgrind
      COMMAND dot -Tsvg ${TARGET}.dot -o ${TARGET}-graph.svg
      DEPENDS ${TARGET}.callgrind)
  endif()
endforeach()
