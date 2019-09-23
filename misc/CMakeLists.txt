if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "-fcolor-diagnostics")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "-fdiagnostics-color=always")
endif()

add_definitions(-DFULLPATH=\"${CMAKE_CURRENT_BINARY_DIR}\")
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
macro(create_executable NAME)
  add_executable (${NAME} ${CMAKE_CURRENT_SOURCE_DIR}/${NAME}.cpp)
  target_link_libraries(${NAME} tatooine)
  target_compile_features(${NAME} PUBLIC cxx_std_17)
  target_compile_options(${NAME} PRIVATE
    -Wall -Wextra -pedantic -Wno-missing-braces -Wno-unused-lambda-capture)
endmacro()

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
create_executable(doublegyre_criticalpointpath_export)
create_executable(newdoublegyre_criticalpointpath_export)
create_executable(newdoublegyre_bifurcationline_export)
create_executable(counterexample_sadlo_bifurcationline_export)

create_executable(counterexample_sadlo_pv_to_streamline)
create_executable(doublegyre_pv_to_streamline)
create_executable(doublegyre_criticalpointpath_to_streamline)
create_executable(newdoublegyre_pv_to_streamline)
create_executable(newdoublegyre_criticalpointpath_to_streamline)
