add_executable(example.direct_volume direct_volume.cpp)
target_link_libraries(example.direct_volume tatooine)
add_custom_target(example.direct_volume.run
                  COMMAND ./example.direct_volume
                  DEPENDS example.direct_volume)

add_executable(example.direct_iso_tornado direct_iso_tornado.cpp)
target_link_libraries(example.direct_iso_tornado tatooine)
add_custom_target(example.direct_iso_tornado.run
                  COMMAND ./example.direct_iso_tornado
                  DEPENDS example.direct_iso_tornado)

add_executable(example.direct_iso_discretized_tornado direct_iso_discretized_tornado.cpp)
target_link_libraries(example.direct_iso_discretized_tornado tatooine)
add_custom_target(example.direct_iso_discretized_tornado.run
                  COMMAND ./example.direct_iso_discretized_tornado
                  DEPENDS example.direct_iso_discretized_tornado)
