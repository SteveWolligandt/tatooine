#===============================================================================
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "-fcolor-diagnostics")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "-fdiagnostics-color=always")
endif()

#===============================================================================
macro(create_executable NAME)
  add_executable (${NAME} ${CMAKE_CURRENT_SOURCE_DIR}/${NAME}.cpp)
  target_link_libraries(${NAME} tatooine)
  target_compile_features(${NAME} PUBLIC cxx_std_17)
  target_compile_options(${NAME} PRIVATE
    -Wall -Wextra -pedantic -Wno-missing-braces -Wno-unused-lambda-capture)
endmacro()

#===============================================================================
create_executable(newdoublegyre_criticalpointpath_export)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
add_executable(newdoublegyre_bifurcationline_export
  newdoublegyre_bifurcationline_export.cpp)
target_link_libraries(newdoublegyre_bifurcationline_export
  PUBLIC tatooine)
target_compile_features(newdoublegyre_bifurcationline_export PUBLIC cxx_std_17)
target_compile_options(newdoublegyre_bifurcationline_export PRIVATE
  -Wall -Wextra -pedantic -Wno-missing-braces -Wno-unused-lambda-capture)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
add_executable(doublegyre_criticalpointpath_export
  doublegyre_criticalpointpath_export.cpp)
target_link_libraries(doublegyre_criticalpointpath_export
  PUBLIC tatooine)
target_compile_features(doublegyre_criticalpointpath_export PUBLIC cxx_std_17)
target_compile_options(doublegyre_criticalpointpath_export PRIVATE
  -Wall -Wextra -pedantic -Wno-missing-braces -Wno-unused-lambda-capture)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
add_executable(counterexample_sadlo_bifurcationline_export
  counterexample_sadlo_bifurcationline_export.cpp)
target_link_libraries(counterexample_sadlo_bifurcationline_export
  PUBLIC tatooine)
target_compile_features(counterexample_sadlo_bifurcationline_export PUBLIC cxx_std_17)
target_compile_options(counterexample_sadlo_bifurcationline_export PRIVATE
  -Wall -Wextra -pedantic -Wno-missing-braces -Wno-unused-lambda-capture)

#===============================================================================
add_subdirectory(bifurcationline)
