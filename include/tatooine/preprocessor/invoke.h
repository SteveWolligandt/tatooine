#ifndef TATOOINE_PREPROCESSOR_INVOKE_H
#define TATOOINE_PREPROCESSOR_INVOKE_H
//==============================================================================
#include <tatooine/preprocessor/expand.h>
#include <tatooine/preprocessor/empty_variadic.h>
#include <tatooine/preprocessor/num_args.h>
#include <tatooine/preprocessor/if.h>
#include <tatooine/preprocessor/equal.h>
//==============================================================================
#define TATOOINE_PP_INVOKE(M, ...) TATOOINE_PP_INVOKE_I(M, ##__VA_ARGS__)
#define TATOOINE_PP_INVOKE_I(M, ...) TATOOINE_PP_EXPAND(M(__VA_ARGS__))
//==============================================================================
#endif
