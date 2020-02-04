if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "-fcolor-diagnostics")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "-fdiagnostics-color=always")
endif()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
set(ALL_BENCH_CPPS main.cpp)
macro(create_benchmark NAME)
  list(APPEND ALL_BENCH_CPPS "${NAME}.cpp")
  add_executable (${NAME}-benchmark ${NAME}.cpp main.cpp)
  target_link_libraries(${NAME}-benchmark tatooine benchmark)
  target_compile_features(${NAME}-benchmark PUBLIC cxx_std_17)
  target_compile_options(${NAME}-benchmark PRIVATE
    -Wall -Wextra -pedantic -Wno-missing-braces -Wno-unused-lambda-capture)
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
  message(STATUS ${FILENAME})
endforeach()

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# create one executable with all benchmarks
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
add_executable (benchmarks main.cpp ${ALL_BENCH_CPPS})
target_link_libraries(benchmarks tatooine benchmark)
target_compile_features(benchmarks PUBLIC cxx_std_17)
target_compile_options(benchmarks PRIVATE -Wall -Wextra -pedantic
  -Wno-missing-braces -Wno-unused-lambda-capture)
