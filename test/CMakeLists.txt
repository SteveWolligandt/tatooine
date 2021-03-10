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
  add_executable (unittests.${NAME} ${NAME}.cpp main.cpp)
  target_link_libraries(unittests.${NAME} tatooine Catch2::Catch2)
  target_include_directories(unittests.${NAME} PRIVATE ${CMAKE_SOURCE_DIR}/external/catch2/include)
  target_compile_features(unittests.${NAME} PUBLIC cxx_std_20)
  target_compile_options(unittests.${NAME} PRIVATE
    -Wall -Wextra -pedantic -Wno-missing-braces -Wno-gnu-zero-variadic-macro-arguments -ftemplate-backtrace-limit=0)
  if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(unittests.${NAME} PRIVATE -Wno-unused-lambda-capture)
  endif()
  #target_compile_definitions(unittests.${NAME} PRIVATE CATCH_CONFIG_FAST_COMPILE)
  add_custom_target (unittests.${NAME}.run COMMAND ./unittests.${NAME} --use-colour=yes DEPENDS unittests.${NAME})
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR
     "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      add_custom_target(unittests.${NAME}.valgrind
                        COMMAND valgrind --log-file="unittests.${NAME}.valgrind" --tool=memcheck --leak-check=yes --show-reachable=yes --track-fds=yes ./unittests.${NAME}
                        DEPENDS unittests.${NAME})
      add_custom_target(unittests.${NAME}.debug
                        COMMAND cgdb ./unittests.${NAME}
                        DEPENDS unittests.${NAME})
    endif()
  endif()
  add_custom_target (unittests.${NAME}.callgrind
                     COMMAND valgrind --tool=callgrind --callgrind-out-file=unittests.${NAME}.callgrind ./unittests.${NAME}
                     DEPENDS unittests.${NAME})
  add_custom_target (unittests.${NAME}.callgrind.kcachegrind
                     COMMAND kcachegrind unittests.${NAME}.callgrind
                     DEPENDS unittests.${NAME}.callgrind)
  add_custom_target (unittests.${NAME}.callgrind.dot
                     COMMAND gprof2dot --format=callgrind --output=unittests.${NAME}.dot unittests.${NAME}.callgrind
                     COMMAND dot -Tsvg unittests.${NAME}.dot -o unittests.${NAME}-graph.svg
                     DEPENDS unittests.${NAME}.callgrind)
  if (TATOOINE_TEST_USE_FAST_MATH)
    target_compile_options(unittests.${NAME} PRIVATE -ffast-math)
  endif()
  if(CLANG_TIDY_EXECUTABLE AND TATOOINE_USE_CLANG_TIDY)
    set_target_properties(unittests.${NAME} PROPERTIES CXX_CLANG_TIDY "${TATOOINE_CLANG_TIDY_CMD}")
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
add_executable (unittests main.cpp ${TATOOINE_TEST_CPPS})
target_link_libraries(unittests tatooine Catch2::Catch2)
target_include_directories(unittests PRIVATE ${CMAKE_SOURCE_DIR}/external/catch2/include)
target_compile_options(unittests PRIVATE -Wall -Wextra -pedantic
  -Wno-missing-braces -Wno-unused-lambda-capture -Wno-gnu-zero-variadic-macro-arguments)
catch_discover_tests(unittests)
add_custom_target (unittests.run COMMAND ./unittests --use-colour=yes DEPENDS unittests)
add_custom_target (unittests.xml COMMAND ./unittests --reporter=xml DEPENDS unittests)
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR
   "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_custom_target (unittests.valgrind
                       COMMAND valgrind --log-file="unittests.valgrind" --tool=memcheck --leak-check=yes --show-reachable=yes --track-fds=yes ./unittests
                       DEPENDS unittests)
  else()
    add_custom_target (unittests.valgrind
                       COMMAND echo "" > /dev/null
                       COMMENT "Target ${BoldRed}unittests.valgrind ${ColourReset}only available in debug mode.")
  endif()
endif()
add_custom_target (unittests.callgrind
                   COMMAND valgrind --tool=callgrind --callgrind-out-file=unittests.callgrind ./unittests
                   DEPENDS unittests)
add_custom_target (unittests.callgrind.kcachegrind
                   COMMAND kcachegrind unittests.callgrind
                   DEPENDS unittests.callgrind)
add_custom_target (unittests.callgrind.dot
                   COMMAND gprof2dot --format=callgrind --output=unittests.dot unittests.callgrind
                   COMMAND dot -Tsvg unittests.dot -o unittests-graph.svg
                   DEPENDS unittests.callgrind)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# COVERAGE
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND (
  "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR
  "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"))
  include(CodeCoverage)
  add_executable (coverage.gcov main.cpp ${TATOOINE_TEST_CPPS})
  set(COVERAGE_EXCLUDES
      "${CMAKE_SOURCE_DIR}/test"
      "${CMAKE_SOURCE_DIR}/external"
      "/usr/include")
  set(COVERAGE_GCOVR_EXCLUDES ${COVERAGE_EXCLUDES})

  SETUP_TARGET_FOR_COVERAGE_GCOVR_HTML(
    NAME coverage.gcov.html
    EXECUTABLE coverage.gcov DEPENDENCIES coverage.gcov)

  SETUP_TARGET_FOR_COVERAGE_GCOVR_XML(
    NAME coverage.gcov.xml
    EXECUTABLE coverage.gcov DEPENDENCIES coverage.gcov)
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_link_libraries(coverage.gcov tatooine Catch2::Catch2 gcov)
    target_compile_options(coverage.gcov
      PRIVATE
        -Wall -Wextra -pedantic
        -fprofile-arcs -ftest-coverage -fno-inline -fno-inline-small-functions -fno-default-inline
        -Wno-missing-braces -Wno-unused-lambda-capture)
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_link_libraries(coverage.gcov gcov tatooine Catch2::Catch2 -fprofile-instr-generate -fcoverage-mapping)
    target_compile_options(coverage.gcov
      PRIVATE
        -Wall -Wextra -pedantic
        -fprofile-arcs -ftest-coverage -fno-inline
        -Wno-missing-braces -Wno-unused-lambda-capture)
    # LLVM Sanitizer
    add_executable (coverage.llvm main.cpp ${TATOOINE_TEST_CPPS})
    target_link_libraries(coverage.llvm tatooine Catch2::Catch2 -fprofile-instr-generate -fcoverage-mapping)
    target_compile_options(coverage.llvm
      PRIVATE
        -Wall -Wextra -pedantic
        -fprofile-instr-generate -fcoverage-mapping
        -Wno-missing-braces -Wno-unused-lambda-capture)

    set(TATOOINE_TEST_COVBIN_PROFRAW ${CMAKE_CURRENT_BINARY_DIR}/coverage.llvm.profraw)
    set(TATTOINE_TEST_COVBIN_PROFDATA ${CMAKE_CURRENT_BINARY_DIR}/coverage.llvm.profdata)
    add_custom_target(coverage.llvm.profraw 
      COMMAND LLVM_PROFILE_FILE=${TATOOINE_TEST_COVBIN_PROFRAW} ./coverage.llvm
      DEPENDS coverage.llvm
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_custom_target(coverage.llvm.profdata 
      COMMAND llvm-profdata merge -sparse ${TATOOINE_TEST_COVBIN_PROFRAW} -o ${TATTOINE_TEST_COVBIN_PROFDATA}
      DEPENDS coverage.llvm.profraw
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_custom_target(coverage.llvm.line 
      COMMAND llvm-cov show ./coverage.llvm -instr-profile=${TATTOINE_TEST_COVBIN_PROFDATA} > coverage.txt
      DEPENDS coverage.llvm.profdata
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_custom_target(coverage.llvm.html 
      COMMAND llvm-cov show ./coverage.llvm -format=html -instr-profile=${TATTOINE_TEST_COVBIN_PROFDATA} > coverage.html
      DEPENDS coverage.llvm.profdata
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  endif()
endif()
endif()
