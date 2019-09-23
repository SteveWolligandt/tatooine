
add_definitions(-DFULLPATH=\"${CMAKE_CURRENT_BINARY_DIR}\")

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
create_executable(counterexample_sadlo_pv_to_streamline)

create_executable(doublegyre_pv_to_streamline)
create_executable(doublegyre_criticalpointpath_to_streamline)
create_executable(doublegyre_straight_line_to_streamline)

create_executable(newdoublegyre_pv_to_streamline)
create_executable(newdoublegyre_criticalpointpath_to_streamline)
create_executable(newdoublegyre_straight_line_to_streamline)

create_executable(straight_line_for_dg)
