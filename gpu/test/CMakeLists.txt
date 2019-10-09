find_package(Catch2 REQUIRED)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "-fcolor-diagnostics")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "-fdiagnostics-color=always")
endif()

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
enable_testing()
include(CTest)
include(Catch)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
set(ALL_TEST_CPPS main.cpp)
macro(create_test NAME)
  set(ALL_TEST_CPPS ${ALL_TEST_CPPS} ${NAME}.cpp)
  add_executable (${NAME} ${NAME}.cpp main.cpp)
  target_link_libraries(${NAME} tatooine tatooine_gpu Catch2::Catch2)
  target_include_directories(${NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../private_include)
  target_compile_features(${NAME} PUBLIC cxx_std_14)
  target_compile_options(${NAME} PRIVATE
    -Wall -Wextra -pedantic -Wno-missing-braces -Wno-unused-lambda-capture)
  catch_discover_tests(${NAME})
endmacro()

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
create_test(add)
create_test(global_buffer)
create_test(texture_buffer)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
add_executable (_allgpu main.cpp ${ALL_TEST_CPPS})
target_link_libraries(_allgpu tatooine Catch2::Catch2)
target_compile_options(_allgpu PRIVATE -Wall -Wextra -pedantic
  -Wno-missing-braces -Wno-unused-lambda-capture)
add_custom_command(TARGET _allgpu POST_BUILD
                   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                   COMMAND mv _allgpu all)
