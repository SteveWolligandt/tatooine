if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  list(APPEND CMAKE_CXX_FLAGS "-fcolor-diagnostics")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  list(APPEND CMAKE_CXX_FLAGS "-fdiagnostics-color=always")
endif()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
add_executable(first_person_window.test first_person_window_test.cpp)
target_link_libraries(first_person_window.test tatooine yavin)
target_compile_features(first_person_window.test PUBLIC cxx_std_17)
add_custom_target(first_person_window.test.run
                   COMMAND ./first_person_window.test
                   DEPENDS first_person_window.test)
