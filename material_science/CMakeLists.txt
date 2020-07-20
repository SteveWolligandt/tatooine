if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  list(APPEND CMAKE_CXX_FLAGS "-fcolor-diagnostics")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  list(APPEND CMAKE_CXX_FLAGS "-fdiagnostics-color=always")
endif()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
function (copy FILE)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${FILE} 
                 ${CMAKE_CURRENT_BINARY_DIR}/${FILE} COPYONLY)
endfunction()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
copy(texture_case1.txt)
copy(texture_case2.txt)
copy(texture_case3.txt)
copy(texture_case4.txt)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
add_executable(material_science.convert_fields convert_fields.cpp)
target_link_libraries(material_science.convert_fields tatooine)
target_compile_features(material_science.convert_fields PUBLIC cxx_std_17)
add_custom_target(material_science.convert_fields.run
                   COMMAND ./material_science.convert_fields
                   DEPENDS material_science.convert_fields)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
add_executable(material_science.find_critical_points find_critical_points.cpp)
target_link_libraries(material_science.find_critical_points tatooine)
target_compile_features(material_science.find_critical_points PUBLIC cxx_std_17)
add_custom_target(material_science.find_critical_points.run
                   COMMAND ./material_science.find_critical_points
                   DEPENDS material_science.find_critical_points
                           material_science.convert_fields.run)
