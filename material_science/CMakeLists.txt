if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  list(APPEND CMAKE_CXX_FLAGS "-fcolor-diagnostics")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  list(APPEND CMAKE_CXX_FLAGS "-fdiagnostics-color=always")
endif()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
function (copy FILE)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${FILE} 
                 ${CMAKE_CURRENT_BINARY_DIR}/${FILE} COPYONLY)
endfunction ()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
copy(texture_case1.txt)
copy(texture_case2.txt)
copy(texture_case3.txt)
copy(texture_case4.txt)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
add_executable(convert_fields convert_fields.cpp)
target_link_libraries(convert_fields tatooine)
target_compile_features(convert_fields PUBLIC cxx_std_17)
copy(texture_case1.txt)
copy(texture_case2.txt)
copy(texture_case3.txt)
copy(texture_case4.txt)
