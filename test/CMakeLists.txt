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
endmacro()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# create tests from cpp files in current directory 
file(GLOB TESTFILES
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)
list(REMOVE_ITEM TESTFILES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
foreach(TESTFILE ${TESTFILES})
  get_filename_component(FILENAME ${TESTFILE} NAME_WE)
  create_test(${FILENAME})
  message(STATUS ${FILENAME})
endforeach()

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# create one executable with all tests
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
add_executable (unittests main.cpp ${ALL_TEST_CPPS})
target_link_libraries(unittests tatooine Catch2::Catch2)
target_compile_options(unittests PRIVATE -Wall -Wextra -pedantic
  -Wno-missing-braces -Wno-unused-lambda-capture)
catch_discover_tests(unittests)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# COVERAGE
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  include(CodeCoverage)
  option(COVERAGE "compile with coverage flags" OFF)
  if (COVERAGE AND NOT CMAKE_COMPILER_IS_GNUCXX)
    message(STATUS
      "code coverage can only be done with g++ compiler at the moment")
  endif()
endif()

if (COVERAGE AND CMAKE_COMPILER_IS_GNUCXX)
  add_executable (covbin main.cpp ${ALL_TEST_CPPS})
  target_link_libraries(covbin tatooine Catch2::Catch2 gcov)
  target_compile_options(covbin
    PRIVATE
      -Wall -Wextra -pedantic
      -fprofile-arcs -ftest-coverage -fno-inline -fno-inline-small-functions -fno-default-inline
      -Wno-missing-braces -Wno-unused-lambda-capture)

  set(EXCLUDES
    "${CMAKE_SOURCE_DIR}/test"
    "${CMAKE_SOURCE_DIR}/external"
    "/usr/include")

  set(COVERAGE_GCOVR_EXCLUDES ${EXCLUDES})
  set(COVERAGE_LCOV_EXCLUDES ${EXCLUDES})

  add_custom_target(coverage 
    COMMAND ./covbin
    DEPENDS covbin
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  #SETUP_TARGET_FOR_COVERAGE_LCOV(
  #  NAME coverage.html
  #  EXECUTABLE covbin DEPENDENCIES covbin)

  SETUP_TARGET_FOR_COVERAGE_GCOVR_HTML(
    NAME coverage.html
    EXECUTABLE covbin DEPENDENCIES covbin)

  SETUP_TARGET_FOR_COVERAGE_GCOVR_XML(
    NAME coverage.xml
    EXECUTABLE covbin DEPENDENCIES covbin)
endif()

