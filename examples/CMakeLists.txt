file(GLOB EXAMPLE_FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)
foreach(EXAMPLE_FILE ${EXAMPLE_FILES})
  get_filename_component(FILENAME ${EXAMPLE_FILE} NAME_WE)
  add_executable(example.${FILENAME} ${FILENAME}.cpp)
  target_link_libraries(example.${FILENAME} PUBLIC tatooine)
  add_custom_target(example.${FILENAME}.run
                    COMMAND ./example.${FILENAME}
                    DEPENDS example.${FILENAME})
endforeach()


if (TATOOINE_GL_AVAILABLE)
  file(GLOB EXAMPLE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/rendering/*.cpp"
  )
  foreach(EXAMPLE_FILE ${EXAMPLE_FILES})
    get_filename_component(FILENAME ${EXAMPLE_FILE} NAME_WE)
    add_executable(example.rendering.${FILENAME} rendering/${FILENAME}.cpp)
    target_link_libraries(example.rendering.${FILENAME} PUBLIC tatooine gl)
    add_custom_target(example.rendering.${FILENAME}.run
                      COMMAND ./example.rendering.${FILENAME}
                      DEPENDS example.rendering.${FILENAME})
  endforeach()
  #
  #add_executable(example.rendering.parallel_coordinates rendering/parallel_coordinates.cpp)
  #target_link_libraries(example.rendering.parallel_coordinates tatooine)
  #add_custom_target(example.rendering.parallel_coordinates.run
  #                  COMMAND ./example.rendering.parallel_coordinates
  #                  DEPENDS example.rendering.parallel_coordinates)
  #
  #add_executable(example.rendering.line rendering/line.cpp)
  #target_link_libraries(example.rendering.line tatooine)
  #add_custom_target(example.rendering.line.run
  #                  COMMAND ./example.rendering.line
  #                  DEPENDS example.rendering.line)
  #
  #add_executable(example.rendering.direct_volume rendering/direct_volume.cpp)
  #target_link_libraries(example.rendering.direct_volume tatooine)
  #add_custom_target(example.rendering.direct_volume.run
  #                  COMMAND ./example.rendering.direct_volume
  #                  DEPENDS example.rendering.direct_volume)
  #
  #add_executable(example.rendering.gl_context rendering/gl_context.cpp)
  #target_link_libraries(example.rendering.gl_context tatooine)
  #add_custom_target(example.rendering.gl_context.run
  #                  COMMAND ./example.rendering.gl_context
  #                  DEPENDS example.rendering.gl_context)
endif()
