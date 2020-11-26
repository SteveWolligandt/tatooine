if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "-fcolor-diagnostics")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "-fdiagnostics-color=always")
endif()
add_executable (smearing.test test.cpp)
target_link_libraries(smearing.test tatooine)
target_compile_features(smearing.test PUBLIC cxx_std_20)
target_compile_options(smearing.test PRIVATE
-Wall -Wextra -pedantic -Wno-missing-braces -Wno-unused-lambda-capture
-march=native)
add_custom_target(smearing.test.run
COMMAND ./smearing.test
DEPENDS smearing.test)
