#ifndef TATOOINE_PREPROCESSOR_IF_HP
#define TATOOINE_PREPROCESSOR_IF_HP
//==============================================================================
#include <tatooine/preprocessor/bool.h>
#include <tatooine/preprocessor/expand.h>
//==============================================================================
#define TATOOINE_PP_IF_ELSE(cond, t, f)                                        \
  TATOOINE_PP_IF_ELSE_BIT(TATOOINE_PP_BOOL(cond), t, f)
//------------------------------------------------------------------------------
#define TATOOINE_PP_IF_ELSE_BIT(bit, t, f)   TATOOINE_PP_IF_ELSE_BIT_I(bit, t, f)
#define TATOOINE_PP_IF_ELSE_BIT_I(bit, t, f) TATOOINE_PP_IF_ELSE_##bit(t, f)
//------------------------------------------------------------------------------
#define TATOOINE_PP_IF_ELSE_0(t, f) f
#define TATOOINE_PP_IF_ELSE_1(t, f) t
//==============================================================================
#define TATOOINE_PP_IF(cond, t) TATOOINE_PP_IF_BIT(TATOOINE_PP_BOOL(cond), (t))
//------------------------------------------------------------------------------
#define TATOOINE_PP_IF_BIT(bit, t)   TATOOINE_PP_IF_BIT_I(bit, t)
#define TATOOINE_PP_IF_BIT_I(bit, t) TATOOINE_PP_IF_##bit(t)
//------------------------------------------------------------------------------
#define TATOOINE_PP_IF_0(t)
#define TATOOINE_PP_IF_1(t) TATOOINE_PP_EXPAND t
//==============================================================================
#endif
