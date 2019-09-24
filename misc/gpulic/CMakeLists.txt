find_package(yavin REQUIRED)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/lic.comp
               ${CMAKE_CURRENT_BINARY_DIR}/lic.comp COPYONLY)

add_executable (lic ${CMAKE_CURRENT_SOURCE_DIR}/lic.cpp)
target_link_libraries(lic tatooine yavin)
target_compile_features(lic PUBLIC cxx_std_17)
target_compile_options(lic PRIVATE
  -Wall -Wextra -pedantic -Wno-missing-braces -Wno-unused-lambda-capture)
