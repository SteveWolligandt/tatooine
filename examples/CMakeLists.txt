add_executable(example.constrained_delaunay constrained_delaunay.cpp)
target_link_libraries(example.constrained_delaunay tatooine)
add_custom_target(example.constrained_delaunay.run
                  COMMAND ./example.constrained_delaunay
                  DEPENDS example.constrained_delaunay)

add_executable(example.naive_flowmap_discretization naive_flowmap_discretization.cpp)
target_link_libraries(example.naive_flowmap_discretization tatooine)
add_custom_target(example.naive_flowmap_discretization.run
                  COMMAND ./example.naive_flowmap_discretization
                  DEPENDS example.naive_flowmap_discretization)

add_executable(example.agranovsky_flowmap_discretization
                       agranovsky_flowmap_discretization.cpp)
target_link_libraries(example.agranovsky_flowmap_discretization tatooine)
add_custom_target(example.agranovsky_flowmap_discretization.run
                  COMMAND ./example.agranovsky_flowmap_discretization
                  DEPENDS example.agranovsky_flowmap_discretization)

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

add_executable(example.cutting_plane_doublegyre cutting_plane_doublegyre.cpp)
target_link_libraries(example.cutting_plane_doublegyre tatooine)
add_custom_target(example.cutting_plane_doublegyre.run
                  COMMAND ./example.cutting_plane_doublegyre
                  DEPENDS example.cutting_plane_doublegyre)

if (TATOOINE_GL_AVAILABLE)
  add_executable(example.rendering.line rendering/line.cpp)
  target_link_libraries(example.rendering.line tatooine)
  add_custom_target(example.rendering.line.run
                    COMMAND ./example.rendering.line
                    DEPENDS example.rendering.line)
                  
  add_executable(example.rendering.direct_volume rendering/direct_volume.cpp)
  target_link_libraries(example.rendering.direct_volume tatooine)
  add_custom_target(example.rendering.direct_volume.run
                    COMMAND ./example.rendering.direct_volume
                    DEPENDS example.rendering.direct_volume)
                  
  add_executable(example.rendering.gl_context rendering/gl_context.cpp)
  target_link_libraries(example.rendering.gl_context tatooine)
  add_custom_target(example.rendering.gl_context.run
                    COMMAND ./example.rendering.gl_context
                    DEPENDS example.rendering.gl_context)
endif()
