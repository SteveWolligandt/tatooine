find_package(Catch2 REQUIRED)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "-fcolor-diagnostics")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "-fdiagnostics-color=always")
endif()

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
enable_testing()
include(CTest)
include(Catch)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
set(ALL_TEST_CPPS main.cpp)
macro(create_test NAME)
  set(ALL_TEST_CPPS ${ALL_TEST_CPPS} ${NAME}.cpp)
  add_executable (${NAME} ${NAME}.cpp main.cpp)
  target_link_libraries(${NAME} tatooine Catch2::Catch2)
  target_compile_features(${NAME} PUBLIC cxx_std_17)
  target_compile_options(${NAME} PRIVATE
    -Wall -Wextra -pedantic -Wno-missing-braces -Wno-unused-lambda-capture)
  catch_discover_tests(${NAME})
endmacro()

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
create_test(doublegyre)
create_test(abcflow)
create_test(tensor)
create_test(integration)
create_test(diff)
create_test(field_operations)
create_test(symbolic_field)
create_test(line)
create_test(parallel_vectors)
create_test(spacetime)
create_test(grid_sampler)
create_test(chunked_data)
create_test(pointset)
create_test(edgeset)
create_test(mesh)
create_test(streamsurface)
create_test(parameterized_surface)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
add_executable (_all main.cpp ${ALL_TEST_CPPS})
target_link_libraries(_all tatooine Catch2::Catch2)
target_compile_options(_all PRIVATE -Wall -Wextra -pedantic
  -Wno-missing-braces -Wno-unused-lambda-capture)
add_custom_command(TARGET _all POST_BUILD
                   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                   COMMAND mv _all all)
