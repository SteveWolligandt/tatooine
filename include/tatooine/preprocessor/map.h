#ifndef TATOOINE_PREPROCESSOR_MAP_H
#define TATOOINE_PREPROCESSOR_MAP_H
//==============================================================================
#include <tatooine/preprocessor/invoke.h>
#include <tatooine/preprocessor/concat.h>
#include <tatooine/preprocessor/expand.h>
#include <tatooine/preprocessor/apply_f.h>
#include <tatooine/preprocessor/apply_f2.h>
#include <tatooine/preprocessor/num_args.h>
//==============================================================================
#define TATOOINE_PP_MAP(f, ...)                                                \
  TATOOINE_PP_INVOKE(TATOOINE_PP_CONCAT(TATOOINE_PP_APPLY_F_,                  \
                                        TATOOINE_PP_NUM_ARGS(__VA_ARGS__)),    \
                     f, TATOOINE_PP_EXPAND(__VA_ARGS__))

#define TATOOINE_PP_MAP2(f, ...)                                               \
  TATOOINE_PP_INVOKE(TATOOINE_PP_CONCAT(TATOOINE_PP_APPLY_F2_,                 \
                                        TATOOINE_PP_NUM_ARGS(__VA_ARGS__)),    \
                     f, TATOOINE_PP_EXPAND(__VA_ARGS__))
//==============================================================================
#endif
