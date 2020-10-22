#ifndef TATOOINE_PREPROCESSOR_EMPTY_VARIADIC_H
#define TATOOINE_PREPROCESSOR_EMPTY_VARIADIC_H
//==============================================================================
#include <tatooine/preprocessor/equal.h>
#include <tatooine/preprocessor/num_args.h>
//==============================================================================
#define TATOOINE_PP_EMPTY_VARIADIC(...)                                        \
  TATOOINE_PP_EMPTY_VARIADIC_I(0, ##__VA_ARGS__)
#define TATOOINE_PP_EMPTY_VARIADIC_I(...)                                      \
  TATOOINE_PP_EQUAL(1, TATOOINE_PP_NUM_ARGS(__VA_ARGS__))
//------------------------------------------------------------------------------
#define TATOOINE_PP_NOT_EMPTY_VARIADIC(...)                                        \
  TATOOINE_PP_NOT_EMPTY_VARIADIC_I(0, ##__VA_ARGS__)
#define TATOOINE_PP_NOT_EMPTY_VARIADIC_I(...)                                      \
  TATOOINE_PP_NOT_EQUAL(1, TATOOINE_PP_NUM_ARGS(__VA_ARGS__))
//==============================================================================
#endif
