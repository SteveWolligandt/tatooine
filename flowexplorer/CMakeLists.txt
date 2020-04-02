if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  list(APPEND CMAKE_CXX_FLAGS "-fcolor-diagnostics")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  list(APPEND CMAKE_CXX_FLAGS "-fdiagnostics-color=always")
endif()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
add_executable(flowexplorer flowexplorer.cpp)
target_link_libraries(flowexplorer tatooine yavin)
target_compile_features(flowexplorer PUBLIC cxx_std_17)
add_custom_target(flowexplorer.run
                  COMMAND ./flowexplorer
                  DEPENDS flowexplorer)
