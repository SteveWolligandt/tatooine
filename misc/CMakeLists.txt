# ==============================================================================
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "-fcolor-diagnostics")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "-fdiagnostics-color=always")
endif()

# ------------------------------------------------------------------------------
macro(create_executable NAME)
  add_executable(${NAME})
  target_link_libraries(${NAME} tatooine)
  set_property(TARGET ${NAME} PROPERTY CXX_STANDARD 20)
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}"
                                                  STREQUAL "Clang")
    target_compile_options(
      ${NAME} PRIVATE -Wall -Wextra -pedantic -Wno-missing-braces
                      -Wno-unused-lambda-capture)
  endif()
endmacro()

# ------------------------------------------------------------------------------
# create tests from cpp files in current directory
file(GLOB APPFILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
foreach(APPFILE ${APPFILES})
  get_filename_component(FILENAME ${APPFILE} NAME_WE)
  create_executable(${FILENAME})
endforeach()
# ------------------------------------------------------------------------------
create_executable(pathlines_for_thomas)
create_executable(newdoublegyre_criticalpointpath_export)
create_executable(newdoublegyre_bifurcationline_export)
create_executable(doublegyre_criticalpointpath_export)
create_executable(counterexample_sadlo_bifurcationline_export)

# ------------------------------------------------------------------------------
add_subdirectory(bifurcationline)
