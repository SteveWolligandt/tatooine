option(TATOOINE_BUILD_TESTS "Tests shall be built?")
if(${TATOOINE_BUILD_TESTS})
  update_git_submodule(external/catch2)
  configure_file(poly_data.vtp poly_data.vtp COPYONLY)
  configure_file(line.vtp line.vtp COPYONLY)

  set(CATCH_BUILD_TESTING
      OFF
      CACHE BOOL "Build SelfTest project")
  set(CATCH_ENABLE_WERROR
      OFF
      CACHE BOOL "Enable all warnings as errors")
  set(CATCH_INSTALL_DOCS
      OFF
      CACHE BOOL "Install documentation alongside library")
  set(CATCH_INSTALL_HELPERS
      OFF
      CACHE BOOL "Install contrib alongside library")
  add_subdirectory(external/catch2 EXCLUDE_FROM_ALL)
  list(APPEND CMAKE_MODULE_PATH
       "${CMAKE_CURRENT_SOURCE_DIR}/external/catch2/contrib/")
  option(TATOOINE_TEST_USE_FAST_MATH "use -ffast-math for unittests" OFF)
  # -----------------------------------------------------------------------------
  enable_testing()
  include(CTest)
  include(Catch)
  include(colors)
  if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND ("${CMAKE_CXX_COMPILER_ID}" STREQUAL
                                            "GNU"))
    include(CodeCoverage)
  endif()
  set(COVERAGE_EXCLUDES)
  set(COVERAGE_GCOVR_EXCLUDES ${COVERAGE_EXCLUDES})
  set(COVERAGE_LCOV_EXCLUDES
      ${CMAKE_SOURCE_DIR}/test/*
      ${CMAKE_SOURCE_DIR}/external/gcem/include/*
      ${CMAKE_SOURCE_DIR}/external/gcem/include/*/*
      ${CMAKE_SOURCE_DIR}/gl/external/imgui/*
      ${CMAKE_SOURCE_DIR}/gl/external/imgui/*/*
      ${CMAKE_SOURCE_DIR}/test/external/catch2/*
      ${CMAKE_SOURCE_DIR}/test/external/catch2/*/*
      ${CMAKE_SOURCE_DIR}/test/external/catch2/*/*/*
      ${CMAKE_SOURCE_DIR}/test/external/catch2/*/*/*/*
      /usr/include/*
      /usr/include/CGAL/*
      /usr/include/CGAL/*/*
      /usr/include/CGAL/*/*/*
      /usr/include/flann/*
      /usr/include/flann/*/*
      /usr/include/flann/*/*/*
      /usr/include/boost
      /usr/include/boost/*
      /usr/include/boost/*/*
      /usr/include/boost/*/*/*
      /usr/include/boost/*/*/*/*
      /usr/include/*/*/*/*/*/*
      /usr/include/png++/*
      /usr/include/*/c++/*/*
      /usr/include/*/c++/*/*/*
      /usr/include/*/c++/*/*/*/*
      /usr/include/c++/*/*
      /usr/include/c++/*/*/*
      /usr/include/c++/*/*/*/*
      /usr/lib/gcc/*
      /usr/lib/gcc/*/*/*
      /usr/lib/gcc/*/*/*
      /usr/lib/gcc/*/*/*/*)
  # ----------------------------------------------------------------------------
  set(TATOOINE_TEST_CPPS main.cpp)
  macro(create_test NAME)
    set(TARGET ${NAME}.unittests)
    list(APPEND TATOOINE_TEST_CPPS "${NAME}.cpp")
    # add and configure unittest-target
    add_executable(${TARGET} EXCLUDE_FROM_ALL ${NAME}.cpp main.cpp)
    target_include_directories(${TARGET}
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    if(TATOOINE_USE_CLANG_TIDY)
      set_target_properties(
        ${TARGET}
        PROPERTIES
          CXX_CLANG_TIDY
          "${CLANG_TIDY_EXECUTABLE};\
           -checks=${CLANG_TIDY_CHECKS};\
           -header-filter='${CMAKE_SOURCE_DIR}/*';\
           -extra-arg=-std=c++20;\
           -line-filter=[{\"name\":\"${NAME}.cpp\",\"lines\":[[1,1]]}]")
    endif()

    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
      target_compile_options(${TARGET} PRIVATE -fcolor-diagnostics)
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
      target_compile_options(${TARGET} PRIVATE -fdiagnostics-color=always)
    endif()

    target_link_libraries(${TARGET} base tensor fields rendering Catch2::Catch2)
    set_property(TARGET ${TARGET} PROPERTY CXX_STANDARD 20)
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}"
                                                    STREQUAL "Clang")
      target_compile_options(
        ${TARGET}
        PRIVATE -Wall -Wextra -pedantic -Wno-missing-braces
                -Wno-gnu-zero-variadic-macro-arguments
                -ftemplate-backtrace-limit=0)
    endif()
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      target_compile_options(${TARGET} PRIVATE -fconcepts-diagnostics-depth=10)
    endif()
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      target_compile_options(${TARGET} PRIVATE -Wno-unused-lambda-capture)
    endif()
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
      target_compile_options(${TARGET} PRIVATE "/W4")
    endif()
    target_compile_definitions(${TARGET} PRIVATE CATCH_CONFIG_FAST_COMPILE)
    if(TATOOINE_TEST_USE_FAST_MATH)
      target_compile_options(${TARGET} PRIVATE -ffast-math)
    endif()

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      # add valgrind targets for unittest
      add_custom_target(
        ${TARGET}.valgrind
        COMMAND
          valgrind --log-file="${TARGET}.valgrind" --tool=memcheck
          --leak-check=yes --show-reachable=yes --track-fds=yes ./${TARGET}
        DEPENDS ${TARGET})
      add_custom_target(
        ${TARGET}.callgrind
        EXCLUDE_FROM_ALL
        COMMAND valgrind --tool=callgrind
                --callgrind-out-file=${TARGET}.callgrind ./${TARGET}
        DEPENDS ${TARGET})
      add_custom_target(
        ${TARGET}.callgrind.kcachegrind
        EXCLUDE_FROM_ALL
        COMMAND kcachegrind ${TARGET}.callgrind
        DEPENDS ${TARGET}.callgrind)
      add_custom_target(
        ${TARGET}.callgrind.dot
        EXCLUDE_FROM_ALL
        COMMAND gprof2dot --format=callgrind --output=${TARGET}.dot
                ${TARGET}.callgrind
        COMMAND dot -Tsvg ${TARGET}.dot -o ${TARGET}-graph.svg
        DEPENDS ${TARGET}.callgrind)

      if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        # code coverage when using gcc in debug
        add_executable(${TARGET}.gcov main.cpp ${NAME}.cpp)
        target_include_directories(${TARGET}.gcov
                                   PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
        if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
          target_compile_options(${TARGET}.gcov PRIVATE -fcolor-diagnostics)
        elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
          target_compile_options(${TARGET}.gcov
                                 PRIVATE -fdiagnostics-color=always)
        endif()
        set_property(TARGET ${TARGET}.gcov PROPERTY CXX_STANDARD 20)
        target_link_libraries(${TARGET}.gcov base fields rendering
                              Catch2::Catch2 gcov)
        target_compile_options(
          ${TARGET}.gcov
          PRIVATE -Wall
                  -Wextra
                  -pedantic
                  -fprofile-arcs
                  -ftest-coverage
                  -fno-inline
                  -fno-inline-small-functions
                  -fno-default-inline
                  -Wno-missing-braces
                  -Wno-unused-lambda-capture)

        if(LCOV_PATH)
          setup_target_for_coverage_lcov(
            NAME ${TARGET}.lcov EXECUTABLE ${TARGET}.gcov DEPENDENCIES
            ${TARGET}.gcov)
        endif()

        setup_target_for_coverage_gcovr_html(
          NAME ${TARGET}.gcovr.html EXECUTABLE ${TARGET}.gcov DEPENDENCIES
          ${TARGET}.gcov)

        setup_target_for_coverage_gcovr_xml(
          NAME ${TARGET}.gcovr.xml EXECUTABLE ${TARGET}.gcov DEPENDENCIES
          ${TARGET}.gcov)
        add_custom_target(
          ${TARGET}.codecoverage
          COMMAND rm -rf ${NAME}_codecoverage
          COMMAND mv ${TARGET}.lcov ${NAME}_codecoverage
          DEPENDS ${TARGET}.lcov)
      endif()
    endif()

  endmacro()
  # * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  #   - - create tests from cpp files in current directory
  file(GLOB TESTFILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
  list(REMOVE_ITEM TESTFILES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
  foreach(TESTFILE ${TESTFILES})
    get_filename_component(FILENAME ${TESTFILE} NAME_WE)
    create_test(${FILENAME})
  endforeach()
  # ----------------------------------------------------------------------------
  # create one executable with all tests
  # ----------------------------------------------------------------------------
  add_executable(unittests main.cpp ${TATOOINE_TEST_CPPS})
  target_include_directories(unittests
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_compile_options(unittests PRIVATE -fcolor-diagnostics)
  elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(unittests PRIVATE -fdiagnostics-color=always)
  endif()
  set_property(TARGET unittests PROPERTY CXX_STANDARD 20)
  if(TATOOINE_USE_CLANG_TIDY)
    set_target_properties(
      unittests
      PROPERTIES
        CXX_CLANG_TIDY
        "${CLANG_TIDY_EXECUTABLE};-checks=${CLANG_TIDY_CHECKS};-header-filter='${CMAKE_SOURCE_DIR}/*'"
    )
  endif()
  target_link_libraries(unittests base fields rendering Catch2::Catch2)
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}"
                                                  STREQUAL "Clang")
    target_compile_options(
      unittests
      PRIVATE -Wall -Wextra -pedantic -Wno-missing-braces
              -Wno-unused-lambda-capture -Wno-gnu-zero-variadic-macro-arguments)
  endif()
  catch_discover_tests(unittests)
  add_custom_target(
    unittests.xml
    COMMAND ./unittests --reporter=xml
    DEPENDS unittests)

  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # add valgrind targets for unittest
    add_custom_target(
      unittests.valgrind
      COMMAND valgrind --log-file="unittests.valgrind" --tool=memcheck
              --leak-check=yes --show-reachable=yes --track-fds=yes ./unittests
      DEPENDS unittests)
    add_custom_target(
      unittests.callgrind
      COMMAND valgrind --tool=callgrind --callgrind-out-file=unittests.callgrind
              ./unittests
      DEPENDS unittests)
    add_custom_target(
      unittests.callgrind.kcachegrind
      COMMAND kcachegrind unittests.callgrind
      DEPENDS unittests.callgrind)
    add_custom_target(
      unittests.callgrind.dot
      COMMAND gprof2dot --format=callgrind --output=unittests.dot
              unittests.callgrind
      COMMAND dot -Tsvg unittests.dot -o unittests-graph.svg
      DEPENDS unittests.callgrind)

    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
      # code coverage when using gcc in debug
      add_executable(unittests.gcov main.cpp ${TATOOINE_TEST_CPPS})
      target_include_directories(unittests.gcov
                                 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
      if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        target_compile_options(unittests.gcov PRIVATE -fcolor-diagnostics)
      elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        target_compile_options(unittests.gcov
                               PRIVATE -fdiagnostics-color=always)
      endif()
      target_link_libraries(unittests.gcov base fields rendering Catch2::Catch2
                            gcov)
      target_compile_options(
        unittests.gcov
        PRIVATE -Wall
                -Wextra
                -pedantic
                -fprofile-arcs
                -ftest-coverage
                -fno-inline
                -fno-inline-small-functions
                -fno-default-inline
                -Wno-missing-braces
                -Wno-unused-lambda-capture)

      if(LCOV_PATH)
        setup_target_for_coverage_lcov(
          NAME unittests.lcov EXECUTABLE unittests.gcov DEPENDENCIES
          unittests.gcov)
      endif()

      setup_target_for_coverage_gcovr_html(
        NAME unittests.gcovr.html EXECUTABLE unittests.gcov DEPENDENCIES
        unittests.gcov)

      setup_target_for_coverage_gcovr_xml(
        NAME unittests.gcovr.xml EXECUTABLE unittests.gcov DEPENDENCIES
        unittests.gcov)
      add_custom_target(
        unittests.codecoverage
        COMMAND rm -rf codecoverage
        COMMAND mv unittests.lcov codecoverage
        DEPENDS unittests.lcov)
    endif()
  endif()
endif()
