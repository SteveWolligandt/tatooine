file(GLOB EXAMPLE_FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)
foreach(EXAMPLE_FILE ${EXAMPLE_FILES})
  get_filename_component(FILENAME ${EXAMPLE_FILE} NAME_WE)
  add_executable(geometry.example.${FILENAME} ${FILENAME}.cpp)
  target_link_libraries(geometry.example.${FILENAME} PUBLIC geometry)
  add_custom_target(geometry.example.${FILENAME}.run
                    COMMAND ./geometry.example.${FILENAME}
                    DEPENDS geometry.example.${FILENAME})
endforeach()
