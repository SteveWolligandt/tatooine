#ifndef TATOOINE_PREPROCESSOR_INVOKE_H
#define TATOOINE_PREPROCESSOR_INVOKE_H
//==============================================================================
#include <tatooine/preprocessor/expand.h>
//==============================================================================
#define TATOOINE_PP_INVOKE(M, ...) TATOOINE_PP_EXPAND(M(__VA_ARGS__))
//==============================================================================
#endif
