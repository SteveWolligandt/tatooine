add_executable(example.direct_volume direct_volume.cpp)
target_link_libraries(example.direct_volume tatooine)
add_custom_target(example.direct_volume.run
                  COMMAND ./example.direct_volume
                  DEPENDS example.direct_volume)

add_executable(example.direct_isosurface_rendering_analytical_tornado direct_isosurface_rendering_analytical_tornado.cpp)
target_link_libraries(example.direct_isosurface_rendering_analytical_tornado tatooine)
add_custom_target(example.direct_isosurface_rendering_analytical_tornado.run
                  COMMAND ./example.direct_isosurface_rendering_analytical_tornado
                  DEPENDS example.direct_isosurface_rendering_analytical_tornado)

add_executable(example.direct_isosurface_rendering_discretized_tornado direct_isosurface_rendering_discretized_tornado.cpp)
target_link_libraries(example.direct_isosurface_rendering_discretized_tornado tatooine)
add_custom_target(example.direct_isosurface_rendering_discretized_tornado.run
                  COMMAND ./example.direct_isosurface_rendering_discretized_tornado
                  DEPENDS example.direct_isosurface_rendering_discretized_tornado)
