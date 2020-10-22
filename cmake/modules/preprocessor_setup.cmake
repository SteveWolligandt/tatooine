################################################################################
# num_args
foreach(i RANGE 1 ${TATOOINE_PP_NUM_MAX_INDICES})
  list(APPEND FORWARD_INDICES ${i})
endforeach()
set(FORWARD_INDICES_WITH_ZERO 0 ${FORWARD_INDICES})
set(BACKWARD_INDICES_WITH_ZERO ${BACKWARD_INDICES} 0)
foreach(i RANGE ${TATOOINE_PP_NUM_MAX_INDICES} 1 -1)
  list(APPEND BACKWARD_INDICES ${i})
endforeach()

string(JOIN ", " FORWARD_INDICES_COMMA ${FORWARD_INDICES})
string(JOIN ", " FORWARD_INDICES_WITH_ZERO_COMMA ${FORWARD_INDICES})
string(JOIN ", " BACKWARD_INDICES_COMMA ${BACKWARD_INDICES})
string(JOIN ", " BACKWARD_INDICES_WITH_ZERO_COMMA ${BACKWARD_INDICES})

string(JOIN ", _" FORWARD_INDICES_UNDERSCORE_COMMA ${FORWARD_INDICES})
string(CONCAT FORWARD_INDICES_UNDERSCORE_COMMA "_" ${FORWARD_INDICES_UNDERSCORE_COMMA})

string(JOIN ", _" FORWARD_INDICES_WITH_ZERO_UNDERSCORE_COMMA ${FORWARD_INDICES_WITH_ZERO})
string(CONCAT FORWARD_INDICES_WITH_ZERO_UNDERSCORE_COMMA "_" ${FORWARD_INDICES_WITH_ZERO_UNDERSCORE_COMMA})

string(JOIN ", _" BACKWARD_INDICES_UNDERSCORE_COMMA ${BACKWARD_INDICES})
string(CONCAT BACKWARD_INDICES_UNDERSCORE_COMMA "_" ${BACKWARD_INDICES_UNDERSCORE_COMMA})

string(JOIN ", _" BACKWARD_INDICES_WITH_ZERO_UNDERSCORE_COMMA ${BACKWARD_INDICES_WITH_ZERO})
string(CONCAT BACKWARD_INDICES_WITH_ZERO_UNDERSCORE_COMMA "_" ${BACKWARD_INDICES_WITH_ZERO_UNDERSCORE_COMMA})

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/preprocessor/num_args.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/tatooine/preprocessor/num_args.h)

################################################################################
# odds definitions
set (INDICES)
set (ODD_INDICES)
set(CMAKE_ODDS "#define TATOOINE_PP_ODDS_0()")
foreach(i RANGE 1 ${TATOOINE_PP_NUM_MAX_INDICES})
  set (INDICES ${INDICES} ${i})
  math(EXPR is_odd "${i} % 2")
  if (${is_odd} EQUAL 1) 
    set (ODD_INDICES ${ODD_INDICES} ${i})
  endif()
  string(JOIN ", _" INDICES_STR ${INDICES})
  string(CONCAT INDICES_STR "_" ${INDICES_STR})
  string(JOIN ", _" ODD_INDICES_STR ${ODD_INDICES})
  string(CONCAT ODD_INDICES_STR "_" ${ODD_INDICES_STR})
  string(CONCAT CMAKE_ODDS ${CMAKE_ODDS} "\n#define TATOOINE_PP_ODDS_${i}(${INDICES_STR}) ${ODD_INDICES_STR}")
endforeach()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/preprocessor/odds.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/tatooine/preprocessor/odds.h)
################################################################################
# evens definitions
set(INDICES)
set(EVEN_INDICES)
set(CMAKE_EVENS "#define TATOOINE_PP_EVENS_0()")
foreach(i RANGE 1 ${TATOOINE_PP_NUM_MAX_INDICES})
  set (INDICES ${INDICES} ${i})
  math(EXPR is_even "(${i} % 2)")
  if (${is_even} EQUAL 0) 
    set (EVEN_INDICES ${EVEN_INDICES} ${i})
  endif()
  string(JOIN ", _" INDICES_STR ${INDICES})
  string(CONCAT INDICES_STR "_" ${INDICES_STR})
  list(LENGTH EVEN_INDICES num_even_indices)
  string(JOIN ", _" EVEN_INDICES_STR ${EVEN_INDICES})
  if (${num_even_indices} GREATER 0)
    string(CONCAT EVEN_INDICES_STR "_" ${EVEN_INDICES_STR})
  endif()
  string(CONCAT CMAKE_EVENS ${CMAKE_EVENS} "\n#define TATOOINE_PP_EVENS_${i}(${INDICES_STR}) ${EVEN_INDICES_STR}")
endforeach()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/preprocessor/evens.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/tatooine/preprocessor/evens.h)
################################################################################
# apply_f definitions
set (INDICES)
set(CMAKE_APPLY_F "#define TATOOINE_PP_APPLY_F_0(f)")
foreach(i RANGE 1 ${TATOOINE_PP_NUM_MAX_INDICES} 1)
  set (INDICES ${INDICES} ${i})
  string(CONCAT CALLS ${CALLS} " f(_${i})")
  string(JOIN ", _" INDICES_STR ${INDICES})
  string(CONCAT INDICES_STR "_" ${INDICES_STR})
  string(CONCAT CMAKE_APPLY_F ${CMAKE_APPLY_F} "\n#define TATOOINE_PP_APPLY_F_${i}(f, ${INDICES_STR}) ${CALLS}")
endforeach()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/preprocessor/apply_f.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/tatooine/preprocessor/apply_f.h)
################################################################################
# apply_f2 definitions
set(INDICES)
set(CALLS)
set(CMAKE_APPLY_F2 "#define TATOOINE_PP_APPLY_F2_0(f)")
foreach(i RANGE 2 ${TATOOINE_PP_NUM_MAX_INDICES} 2)
  math(EXPR j "${i}-1")
  set (INDICES ${INDICES} ${j} ${i})
  string(CONCAT CALLS ${CALLS} " f(_${j}, _${i})")
  string(JOIN ", _" INDICES_STR ${INDICES})
  string(CONCAT INDICES_STR "_" ${INDICES_STR})
  string(CONCAT CMAKE_APPLY_F2 ${CMAKE_APPLY_F2} "\n#define TATOOINE_PP_APPLY_F2_${i}(f, ${INDICES_STR}) ${CALLS}")
endforeach()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/preprocessor/apply_f2.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/tatooine/preprocessor/apply_f2.h)
