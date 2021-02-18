#===============================================================================
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "-fcolor-diagnostics")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "-fdiagnostics-color=always")
endif()
#===============================================================================
set(TATOOINE_SCIVIS_CONTEST_2020_ENSEMBLE_FILE_LIST_ROOT ""
  CACHE PATH "Root path to ensemble members")
#===============================================================================
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ensemble_file_paths.cpp.in
               ${CMAKE_CURRENT_BINARY_DIR}/ensemble_file_paths.cpp)
  #===============================================================================
macro(create_executable NAME)
  add_executable (${NAME}.sciviscontest2020
                  ${NAME}.cpp
                  ${CMAKE_CURRENT_BINARY_DIR}/ensemble_file_paths.cpp)
  target_link_libraries(${NAME}.sciviscontest2020 tatooine)
  target_compile_options(${NAME}.sciviscontest2020 PRIVATE
    -Wall -Wextra -pedantic -Wno-missing-braces -Wno-unused-lambda-capture -march=native)
endmacro()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# create tests from cpp files in current directory 
file(GLOB APPFILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)
foreach(APPFILE ${APPFILES})
  get_filename_component(FILENAME ${APPFILE} NAME_WE)
  create_executable(${FILENAME})
endforeach()
