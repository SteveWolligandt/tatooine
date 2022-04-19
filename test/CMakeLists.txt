option(TATOOINE_BUILD_TESTS "Tests shall be built?")
if(${TATOOINE_BUILD_TESTS})
update_git_submodule(external/catch2)

set(CATCH_BUILD_TESTING OFF CACHE BOOL "Build SelfTest project")
set(CATCH_ENABLE_WERROR OFF CACHE BOOL "Enable all warnings as errors")
set(CATCH_INSTALL_DOCS OFF CACHE BOOL "Install documentation alongside library")
set(CATCH_INSTALL_HELPERS OFF CACHE BOOL "Install contrib alongside library")
add_subdirectory(external/catch2 EXCLUDE_FROM_ALL)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/external/catch2/contrib/")
option(TATOOINE_TEST_USE_FAST_MATH
    "use -ffast-math for unittests" OFF)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
list(APPEND CMAKE_CXX_FLAGS "-fcolor-diagnostics")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  list(APPEND CMAKE_CXX_FLAGS "-fdiagnostics-color=always")
endif()
enable_testing()
include(CTest)
include(Catch)
include(colors)
if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND (
  "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU"
  #OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"
  ))
  include(CodeCoverage)
endif()
set(COVERAGE_EXCLUDES
    ${CMAKE_SOURCE_DIR}/test
    ${CMAKE_SOURCE_DIR}/external/
    ${CMAKE_CURRENT_SOURCE_DIR}/external
    /usr/include)
set(COVERAGE_GCOVR_EXCLUDES ${COVERAGE_EXCLUDES})
set(COVERAGE_LCOV_EXCLUDES
  /usr/include/boost/*/*
  /usr/include/boost/*/*/*
  /usr/include/png++/*
  /usr/include/c++/*/*
  /usr/include/c++/*/*/*
  /usr/include/c++/*/*/*/*
  ${CMAKE_CURRENT_SOURCE_DIR}/*
  ${CMAKE_CURRENT_SOURCE_DIR}/external/catch2/single_include/catch2/*)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
set(TATOOINE_TEST_CPPS main.cpp)
macro(create_test NAME)
  list(APPEND TATOOINE_TEST_CPPS "${NAME}.cpp")
  # add and configure unittest-target
  add_executable(
    ${NAME}.unittests EXCLUDE_FROM_ALL
    ${NAME}.cpp main.cpp)
  if(TATOOINE_USE_CLANG_TIDY)
    set_target_properties(
      ${NAME}.unittests
      PROPERTIES CXX_CLANG_TIDY
      "${CLANG_TIDY_EXECUTABLE};-checks=${CLANG_TIDY_CHECKS};-header-filter='${CMAKE_SOURCE_DIR}/*';-extra-arg=-std=c++17;-line-filter=[{\"name\":\"${NAME}.cpp\",\"lines\":[[1,1]]}]")
  endif()
  target_link_libraries(
    ${NAME}.unittests
    tatooine Catch2::Catch2)
  target_include_directories(
    ${NAME}.unittests
    PRIVATE
      ${CMAKE_SOURCE_DIR}/external/catch2/include)
  target_compile_features(
    ${NAME}.unittests
      PUBLIC
        cxx_std_20)
  target_compile_options(
    ${NAME}.unittests
    PRIVATE
      -Wall -Wextra -pedantic -Wno-missing-braces
      -Wno-gnu-zero-variadic-macro-arguments -ftemplate-backtrace-limit=0)
  if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(
      ${NAME}.unittests
      PRIVATE
        -Wno-unused-lambda-capture)
  endif()
  target_compile_definitions(
    ${NAME}.unittests
    PRIVATE
      CATCH_CONFIG_FAST_COMPILE)
  add_custom_target(
    ${NAME}.unittests.run
    COMMAND ./${NAME}.unittests
            --use-colour=yes
    DEPENDS ${NAME}.unittests)
  if (TATOOINE_TEST_USE_FAST_MATH)
    target_compile_options(
      ${NAME}.unittests
      PRIVATE
        -ffast-math)
  endif()

  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # add valgrind targets for unittest
    add_custom_target(
      ${NAME}.unittests.valgrind
     COMMAND valgrind
             --log-file="${NAME}.unittests.valgrind"
             --tool=memcheck
             --leak-check=yes
             --show-reachable=yes
             --track-fds=yes
             ./${NAME}.unittests
      DEPENDS ${NAME}.unittests)
    add_custom_target (
      ${NAME}.unittests.callgrind
      COMMAND valgrind
              --tool=callgrind
              --callgrind-out-file=${NAME}.unittests.callgrind
              ./${NAME}.unittests
      DEPENDS ${NAME}.unittests)
    add_custom_target (
      ${NAME}.unittests.callgrind.kcachegrind
      COMMAND kcachegrind
              ${NAME}.unittests.callgrind
      DEPENDS ${NAME}.unittests.callgrind)
    add_custom_target (
      ${NAME}.unittests.callgrind.dot
      COMMAND gprof2dot
              --format=callgrind
              --output=${NAME}.unittests.dot
              ${NAME}.unittests.callgrind
      COMMAND dot
              -Tsvg ${NAME}.unittests.dot
              -o ${NAME}.unittests-graph.svg
      DEPENDS ${NAME}.unittests.callgrind)


    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
      # code coverage when using gcc in debug
      add_executable(
        ${NAME}.unittests.gcov EXCLUDE_FROM_ALL
        main.cpp ${NAME}.cpp)
      target_link_libraries(
        ${NAME}.unittests.gcov
        tatooine gl Catch2::Catch2 gcov)
      target_compile_options(
        ${NAME}.unittests.gcov
        PRIVATE
          -Wall -Wextra -pedantic
          -fprofile-arcs -ftest-coverage -fno-inline
          -fno-inline-small-functions -fno-default-inline
          -Wno-missing-braces -Wno-unused-lambda-capture)

      if(LCOV_PATH)
        SETUP_TARGET_FOR_COVERAGE_LCOV(
          NAME ${NAME}.unittests.lcov
          EXECUTABLE ${NAME}.unittests.gcov
          DEPENDENCIES ${NAME}.unittests.gcov)
      endif()

      SETUP_TARGET_FOR_COVERAGE_GCOVR_HTML(
        NAME ${NAME}.unittests.gcovr.html
        EXECUTABLE ${NAME}.unittests.gcov
        DEPENDENCIES ${NAME}.unittests.gcov)

      SETUP_TARGET_FOR_COVERAGE_GCOVR_XML(
        NAME ${NAME}.unittests.gcovr.xml
        EXECUTABLE ${NAME}.unittests.gcov
        DEPENDENCIES ${NAME}.unittests.gcov)
      add_custom_target(
        ${NAME}.unittests.codecoverage
        COMMAND rm -rf ${NAME}_codecoverage
        COMMAND mv ${NAME}.unittests.lcov ${NAME}_codecoverage
        DEPENDS ${NAME}.unittests.lcov)
    endif()
  endif()


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
endforeach()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# create one executable with all tests
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
add_executable (
  unittests
  main.cpp
  ${TATOOINE_TEST_CPPS})
  if(TATOOINE_USE_CLANG_TIDY)
    set_target_properties(
      unittests
      PROPERTIES CXX_CLANG_TIDY
                 "${CLANG_TIDY_EXECUTABLE};-checks=${CLANG_TIDY_CHECKS};-header-filter='${CMAKE_SOURCE_DIR}/*'")
  endif()
target_link_libraries(
  unittests
  tatooine gl Catch2::Catch2)
target_include_directories(
  unittests
  PRIVATE
    ${CMAKE_SOURCE_DIR}/external/catch2/include)
target_compile_options(
  unittests
  PRIVATE
    -Wall -Wextra -pedantic -Wno-missing-braces -Wno-unused-lambda-capture
    -Wno-gnu-zero-variadic-macro-arguments)
catch_discover_tests(unittests)
add_custom_target(
  unittests.run
  COMMAND ./unittests
          --use-colour=yes
  DEPENDS unittests)
add_custom_target (
  unittests.xml
  COMMAND ./unittests
          --reporter=xml
  DEPENDS unittests)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  # add valgrind targets for unittest
  add_custom_target(
    unittests.valgrind
   COMMAND valgrind
           --log-file="unittests.valgrind"
           --tool=memcheck
           --leak-check=yes
           --show-reachable=yes
           --track-fds=yes
           ./unittests
    DEPENDS unittests)
  add_custom_target (
    unittests.callgrind
    COMMAND valgrind
            --tool=callgrind
            --callgrind-out-file=unittests.callgrind
            ./unittests
    DEPENDS unittests)
  add_custom_target (
    unittests.callgrind.kcachegrind
    COMMAND kcachegrind
            unittests.callgrind
    DEPENDS unittests.callgrind)
  add_custom_target (
    unittests.callgrind.dot
    COMMAND gprof2dot
            --format=callgrind
            --output=unittests.dot
            unittests.callgrind
    COMMAND dot
            -Tsvg unittests.dot
            -o unittests-graph.svg
    DEPENDS unittests.callgrind)


  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # code coverage when using gcc in debug
    add_executable(
      unittests.gcov
      main.cpp ${TATOOINE_TEST_CPPS})
    target_link_libraries(
      unittests.gcov
      tatooine gl Catch2::Catch2 gcov)
    target_compile_options(
      unittests.gcov
      PRIVATE
        -Wall -Wextra -pedantic
        -fprofile-arcs -ftest-coverage -fno-inline
        -fno-inline-small-functions -fno-default-inline
        -Wno-missing-braces -Wno-unused-lambda-capture)

    if(LCOV_PATH)
      SETUP_TARGET_FOR_COVERAGE_LCOV(
        NAME unittests.lcov
        EXECUTABLE unittests.gcov
        DEPENDENCIES unittests.gcov)
    endif()

    SETUP_TARGET_FOR_COVERAGE_GCOVR_HTML(
      NAME unittests.gcovr.html
      EXECUTABLE unittests.gcov
      DEPENDENCIES unittests.gcov)

    SETUP_TARGET_FOR_COVERAGE_GCOVR_XML(
      NAME unittests.gcovr.xml
      EXECUTABLE unittests.gcov
      DEPENDENCIES unittests.gcov)
    add_custom_target(
      unittests.codecoverage
      COMMAND rm -rf codecoverage
      COMMAND mv unittests.lcov codecoverage
      DEPENDS unittests.lcov)
  endif()
endif()
endif()
