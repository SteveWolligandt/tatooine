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
  list(APPEND ALL_TEST_CPPS "${NAME}.cpp")
  add_executable (${NAME} ${NAME}.cpp main.cpp)
  target_link_libraries(${NAME} tatooine Catch2::Catch2)
  target_compile_features(${NAME} PUBLIC cxx_std_17)
  target_compile_options(${NAME} PRIVATE
    -Wall -Wextra -pedantic -Wno-missing-braces -Wno-unused-lambda-capture)
  catch_discover_tests(${NAME})
endmacro()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# create tests from cpp files in current directory 
file(GLOB_RECURSE TESTFILES
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)
list(REMOVE_ITEM TESTFILES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
foreach(TESTFILE ${TESTFILES})
  get_filename_component(FILENAME ${TESTFILE} NAME_WE)
  create_test(${FILENAME})
endforeach()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# create one executable with all tests
add_executable (_all main.cpp ${ALL_TEST_CPPS})
target_link_libraries(_all tatooine Catch2::Catch2)
target_compile_options(_all PRIVATE -Wall -Wextra -pedantic
  -Wno-missing-braces -Wno-unused-lambda-capture)
add_custom_command(TARGET _all POST_BUILD
                   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                   COMMAND mv _all all)
