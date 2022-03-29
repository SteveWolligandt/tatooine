file(GLOB EXAMPLE_FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)
foreach(EXAMPLE_FILE ${EXAMPLE_FILES})
  get_filename_component(FILENAME ${EXAMPLE_FILE} NAME_WE)
  add_executable(${FILENAME}.example ${FILENAME}.cpp)
  target_link_libraries(${FILENAME}.example PUBLIC tatooine gl)
  add_custom_target(${FILENAME}.example.run
                    COMMAND ./${FILENAME}.example
                    DEPENDS ${FILENAME}.example)
endforeach()


if (TATOOINE_GL_AVAILABLE)
  file(GLOB EXAMPLE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/rendering/*.cpp"
  )
  foreach(EXAMPLE_FILE ${EXAMPLE_FILES})
    get_filename_component(FILENAME ${EXAMPLE_FILE} NAME_WE)
    add_executable(rendering.${FILENAME}.example rendering/${FILENAME}.cpp)
    target_link_libraries(rendering.${FILENAME}.example PUBLIC tatooine gl)
    add_custom_target(rendering.${FILENAME}.example.run
                      COMMAND ./rendering.${FILENAME}.example
                      DEPENDS rendering.${FILENAME}.example)
  endforeach()
endif()
