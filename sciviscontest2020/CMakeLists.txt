#===============================================================================
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "-fcolor-diagnostics")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "-fdiagnostics-color=always")
endif()
#===============================================================================
macro(create_executable NAME)
  add_executable (${NAME}.sciviscontest2020 ${NAME}.cpp)
  target_link_libraries(${NAME}.sciviscontest2020 tatooine)
  target_compile_features(${NAME}.sciviscontest2020 PUBLIC cxx_std_20)
  target_compile_options(${NAME}.sciviscontest2020 PRIVATE
    -Wall -Wextra -pedantic -Wno-missing-braces -Wno-unused-lambda-capture)
endmacro()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# create tests from cpp files in current directory 
file(GLOB APPFILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)
foreach(APPFILE ${APPFILES})
  get_filename_component(FILENAME ${APPFILE} NAME_WE)
  create_executable(${FILENAME})
endforeach()
