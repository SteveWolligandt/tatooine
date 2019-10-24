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
set(ALL_TEST_CPPS main.cu)
macro(create_test NAME)
  add_executable (${NAME} ${NAME}.cu main.cu)
  set(ALL_TEST_CPPS ${ALL_TEST_CPPS} ${NAME}.cu)
  target_link_libraries(${NAME} tatooine_cuda Catch2::Catch2)
  target_compile_features(${NAME} PUBLIC cxx_std_14)
  target_include_directories(${NAME}
   PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../private_include)
  catch_discover_tests(${NAME})
endmacro()

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
create_test(array)
create_test(buffer)
create_test(tex)
create_test(field_to_tex)

add_executable (cuda_lic lic.cu main.cu)
set(ALL_TEST_CPPS ${ALL_TEST_CPPS} lic.cu)
target_link_libraries(cuda_lic tatooine_cuda Catch2::Catch2)
target_compile_features(cuda_lic PUBLIC cxx_std_14)
target_include_directories(cuda_lic
 PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../private_include)
catch_discover_tests(cuda_lic)
