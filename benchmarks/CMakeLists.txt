if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "-fcolor-diagnostics")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "-fdiagnostics-color=always")
endif()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
set(ALL_BENCH_CPPS main.cpp)
macro(create_benchmark NAME)
  list(APPEND ALL_BENCH_CPPS "${NAME}.cpp")
  add_executable (${NAME}.benchmark ${NAME}.cpp main.cpp)
  target_link_libraries(${NAME}.benchmark tatooine benchmark)
  target_compile_options(${NAME}.benchmark PRIVATE
    -Wall -Wextra -pedantic -Wno-missing-braces -Wno-unused-lambda-capture)
  add_custom_target (${NAME}.benchmark.run COMMAND ./${NAME}.benchmark DEPENDS ${NAME}.benchmark)
  if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND (
    "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR
    "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"))
    add_custom_target (${NAME}.benchmark.valgrind
                       COMMAND valgrind --log-file="${NAME}.benchmark.valgrind" --tool=memcheck --leak-check=yes --show-reachable=yes --track-fds=yes ./${NAME}.benchmark
                       DEPENDS ${NAME}.benchmark)
  endif()
  add_custom_target (${NAME}.benchmark.csv
                     COMMAND ./${NAME}.benchmark --benchmark_out=${NAME}.benchmark.csv --benchmark_out_format=csv --v=0
                     DEPENDS ${NAME}.benchmark)
  add_custom_target (${NAME}.benchmark.json
                     COMMAND ./${NAME}.benchmark --benchmark_out=${NAME}.benchmark.json --benchmark_out_format=json
                     DEPENDS ${NAME}.benchmark)
  add_custom_target (${NAME}.benchmark.callgrind
                     COMMAND valgrind --tool=callgrind --callgrind-out-file=${NAME}.benchmark.callgrind ./${NAME}.benchmark
                     DEPENDS ${NAME}.benchmark)
  add_custom_target (${NAME}.benchmark.callgrind.dot
                     COMMAND gprof2dot --format=callgrind --output=${NAME}.benchmark.dot ${NAME}.benchmark.callgrind
                     COMMAND dot -Tsvg ${NAME}.benchmark.dot -o ${NAME}.benchmark-graph.svg
                     DEPENDS ${NAME}.benchmark.callgrind)
endmacro()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# create benchmarks from cpp files in current directory 
file(GLOB BENCHFILES
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)
list(REMOVE_ITEM BENCHFILES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
foreach(BENCHFILE ${BENCHFILES})
  get_filename_component(FILENAME ${BENCHFILE} NAME_WE)
  create_benchmark(${FILENAME})
endforeach()

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# create one executable with all benchmarks
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
add_executable (benchmarks main.cpp ${ALL_BENCH_CPPS})
target_link_libraries(benchmarks tatooine benchmark)
target_compile_options(benchmarks PRIVATE -Wall -Wextra -pedantic
  -Wno-missing-braces -Wno-unused-lambda-capture)
add_custom_target (benchmarks.run COMMAND ./benchmarks DEPENDS benchmarks)
if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND (
  "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR
  "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"))
  add_custom_target (benchmarks.valgrind
                     COMMAND valgrind --log-file="benchmarks.valgrind" --tool=memcheck --leak-check=yes --show-reachable=yes --track-fds=yes ./benchmarks
                     DEPENDS benchmarks)
endif()
add_custom_target (benchmarks.csv
                   COMMAND ./benchmarks --benchmark_out=benchmarks.csv --benchmark_out_format=csv
                   DEPENDS benchmarks)
add_custom_target (benchmarks.json
                   COMMAND ./benchmarks --benchmark_out=benchmarks.json --benchmark_out_format=json
                   DEPENDS benchmarks)
add_custom_target (benchmarks.callgrind
                   COMMAND valgrind --tool=callgrind --callgrind-out-file=benchmarks.callgrind ./benchmarks
                   DEPENDS benchmarks)
add_custom_target (benchmarks.callgrind.dot
                   COMMAND gprof2dot --format=callgrind --output=benchmarks.dot benchmarks.callgrind
                   COMMAND dot -Tsvg benchmarks.dot -o benchmarks-graph.svg
                   DEPENDS benchmarks.callgrind)
