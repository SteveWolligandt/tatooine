option(TATOOINE_BUILD_TESTS "Tests shall be built?")
if(${TATOOINE_BUILD_TESTS})
update_git_submodule(external/catch2)
add_subdirectory(external/catch2 EXCLUDE_FROM_ALL)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/external/catch2/contrib/")
set(TATOOINE_TEST_USE_FAST_MATH false CACHE BOOL
    "use -ffast-math for unittests")
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
  /usr/include/c++/11/*
  /usr/include/c++/11/bits/*
  /usr/include/c++/11/ext/*
  /usr/include/boost/core/*
  /usr/include/x86_64-linux-gnu/c++/11/bits/*
  /usr/include/png++/*
  /usr/include/boost/system/detail/*
  /usr/include/boost/system/*
  /usr/include/boost/serialization/*
  /usr/include/boost/iterator/*
  ${CMAKE_CURRENT_SOURCE_DIR}/*
  ${CMAKE_CURRENT_SOURCE_DIR}/external/catch2/single_include/catch2/*)
message(STATUS ${CMAKE_CURRENT_SOURCE_DIR}/*)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if(CLANG_TIDY_EXECUTABLE)
  set(TATOOINE_CLANG_TIDY_CMD
    ${CLANG_TIDY_EXECUTABLE}
    ${TATOOINE_CLANG_TIDY_CHECKS}
    "-header-filter=tatooine")
endif()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
set(TATOOINE_TEST_CPPS main.cpp)
macro(create_test NAME)
  list(APPEND TATOOINE_TEST_CPPS "${NAME}.cpp")
  # add and configure unittest-target
  add_executable(
    unittests.${NAME} EXCLUDE_FROM_ALL
    ${NAME}.cpp main.cpp)
  target_link_libraries(
    unittests.${NAME}
    tatooine Catch2::Catch2)
  target_include_directories(
    unittests.${NAME}
    PRIVATE
      ${CMAKE_SOURCE_DIR}/external/catch2/include)
  target_compile_features(
    unittests.${NAME}
      PUBLIC
        cxx_std_20)
  target_compile_options(
    unittests.${NAME}
    PRIVATE
      -Wall -Wextra -pedantic -Wno-missing-braces
      -Wno-gnu-zero-variadic-macro-arguments -ftemplate-backtrace-limit=0)
  if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(
      unittests.${NAME}
      PRIVATE
        -Wno-unused-lambda-capture)
  endif()
  target_compile_definitions(
    unittests.${NAME}
    PRIVATE
      CATCH_CONFIG_FAST_COMPILE)
  add_custom_target(
    unittests.${NAME}.run
    COMMAND ./unittests.${NAME}
            --use-colour=yes
    DEPENDS unittests.${NAME})
  if (TATOOINE_TEST_USE_FAST_MATH)
    target_compile_options(
      unittests.${NAME}
      PRIVATE
        -ffast-math)
  endif()
  if(CLANG_TIDY_EXECUTABLE AND TATOOINE_USE_CLANG_TIDY)
    set_target_properties(
      unittests.${NAME}
      PROPERTIES CXX_CLANG_TIDY
                 "${TATOOINE_CLANG_TIDY_CMD}")
  endif()

  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # add valgrind targets for unittest
    add_custom_target(
      unittests.${NAME}.valgrind
     COMMAND valgrind
             --log-file="unittests.${NAME}.valgrind"
             --tool=memcheck
             --leak-check=yes
             --show-reachable=yes
             --track-fds=yes
             ./unittests.${NAME}
      DEPENDS unittests.${NAME})
    add_custom_target (
      unittests.${NAME}.callgrind
      COMMAND valgrind
              --tool=callgrind
              --callgrind-out-file=unittests.${NAME}.callgrind
              ./unittests.${NAME}
      DEPENDS unittests.${NAME})
    add_custom_target (
      unittests.${NAME}.callgrind.kcachegrind
      COMMAND kcachegrind
              unittests.${NAME}.callgrind
      DEPENDS unittests.${NAME}.callgrind)
    add_custom_target (
      unittests.${NAME}.callgrind.dot
      COMMAND gprof2dot
              --format=callgrind
              --output=unittests.${NAME}.dot
              unittests.${NAME}.callgrind
      COMMAND dot
              -Tsvg unittests.${NAME}.dot
              -o unittests.${NAME}-graph.svg
      DEPENDS unittests.${NAME}.callgrind)


    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
      # code coverage when using gcc in debug
      add_executable(
        unittests.${NAME}.gcov EXCLUDE_FROM_ALL
        main.cpp ${NAME}.cpp)
      target_link_libraries(
        unittests.${NAME}.gcov
        tatooine Catch2::Catch2 gcov)
      target_compile_options(
        unittests.${NAME}.gcov
        PRIVATE
          -Wall -Wextra -pedantic
          -fprofile-arcs -ftest-coverage -fno-inline
          -fno-inline-small-functions -fno-default-inline
          -Wno-missing-braces -Wno-unused-lambda-capture)

      SETUP_TARGET_FOR_COVERAGE_LCOV(
        NAME unittests.${NAME}.lcov
        EXECUTABLE unittests.${NAME}.gcov
        DEPENDENCIES unittests.${NAME}.gcov)

      SETUP_TARGET_FOR_COVERAGE_GCOVR_HTML(
        NAME unittests.${NAME}.gcovr.html
        EXECUTABLE unittests.${NAME}.gcov
        DEPENDENCIES unittests.${NAME}.gcov)

      SETUP_TARGET_FOR_COVERAGE_GCOVR_XML(
        NAME unittests.${NAME}.gcovr.xml
        EXECUTABLE unittests.${NAME}.gcov
        DEPENDENCIES unittests.${NAME}.gcov)
      add_custom_target(
        unittests.${NAME}.codecoverage
        COMMAND rm -rf ${NAME}_codecoverage
        COMMAND mv unittests.${NAME}.lcov ${NAME}_codecoverage
        DEPENDS unittests.${NAME}.lcov)
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
  main.cpp ${TATOOINE_TEST_CPPS})
target_link_libraries(
  unittests
  tatooine Catch2::Catch2)
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
      tatooine Catch2::Catch2 gcov)
    target_compile_options(
      unittests.gcov
      PRIVATE
        -Wall -Wextra -pedantic
        -fprofile-arcs -ftest-coverage -fno-inline
        -fno-inline-small-functions -fno-default-inline
        -Wno-missing-braces -Wno-unused-lambda-capture)

    SETUP_TARGET_FOR_COVERAGE_LCOV(
      NAME unittests.lcov
      EXECUTABLE unittests.gcov
      DEPENDENCIES unittests.gcov)

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
