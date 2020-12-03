if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "-fcolor-diagnostics")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "-fdiagnostics-color=always")
endif()
add_executable (smearing smearing.cpp io.cpp parse_arguments.cpp isolines.cpp)
target_link_libraries(smearing tatooine Boost::program_options)
target_compile_features(smearing PUBLIC cxx_std_20)
target_compile_options(smearing PRIVATE
-Wall -Wextra -pedantic -Wno-missing-braces -Wno-unused-lambda-capture
-march=native)
