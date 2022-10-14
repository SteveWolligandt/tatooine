#ifndef TATOOINE_GL_GLFW_KEYS_H
#define TATOOINE_GL_GLFW_KEYS_H
//==============================================================================
#include <tatooine/gl/glincludes.h>
#include <tatooine/gl/keyboard.h>
//==============================================================================
namespace tatooine::gl {
//==============================================================================
inline auto convert_key(int key) {
  switch (key) {
    case GLFW_KEY_0: return key::KEY_0;
    case GLFW_KEY_1: return key::KEY_1;
    case GLFW_KEY_2: return key::KEY_2;
    case GLFW_KEY_3: return key::KEY_3;
    case GLFW_KEY_4: return key::KEY_4;
    case GLFW_KEY_5: return key::KEY_5;
    case GLFW_KEY_6: return key::KEY_6;
    case GLFW_KEY_7: return key::KEY_7;
    case GLFW_KEY_8: return key::KEY_8;
    case GLFW_KEY_9: return key::KEY_9;
    case GLFW_KEY_F1: return key::KEY_F1;
    case GLFW_KEY_F2: return key::KEY_F2;
    case GLFW_KEY_F3: return key::KEY_F3;
    case GLFW_KEY_F4: return key::KEY_F4;
    case GLFW_KEY_F5: return key::KEY_F5;
    case GLFW_KEY_F6: return key::KEY_F6;
    case GLFW_KEY_F7: return key::KEY_F7;
    case GLFW_KEY_F8: return key::KEY_F8;
    case GLFW_KEY_F9: return key::KEY_F9;
    case GLFW_KEY_F10: return key::KEY_F10;
    case GLFW_KEY_F11: return key::KEY_F11;
    case GLFW_KEY_F12: return key::KEY_F12;
    case GLFW_KEY_F13: return key::KEY_F13;
    case GLFW_KEY_F14: return key::KEY_F14;
    case GLFW_KEY_F15: return key::KEY_F15;
    case GLFW_KEY_F16: return key::KEY_F16;
    case GLFW_KEY_F17: return key::KEY_F17;
    case GLFW_KEY_F18: return key::KEY_F18;
    case GLFW_KEY_F19: return key::KEY_F19;
    case GLFW_KEY_F20: return key::KEY_F20;
    case GLFW_KEY_F21: return key::KEY_F21;
    case GLFW_KEY_F22: return key::KEY_F22;
    case GLFW_KEY_F23: return key::KEY_F23;
    case GLFW_KEY_F24: return key::KEY_F24;
    case GLFW_KEY_F25: return key::KEY_F25;
    case GLFW_KEY_A: return key::KEY_A;
    case GLFW_KEY_B: return key::KEY_B;
    case GLFW_KEY_C: return key::KEY_C;
    case GLFW_KEY_D: return key::KEY_D;
    case GLFW_KEY_E: return key::KEY_E;
    case GLFW_KEY_F: return key::KEY_F;
    case GLFW_KEY_G: return key::KEY_G;
    case GLFW_KEY_H: return key::KEY_H;
    case GLFW_KEY_I: return key::KEY_I;
    case GLFW_KEY_J: return key::KEY_J;
    case GLFW_KEY_K: return key::KEY_K;
    case GLFW_KEY_L: return key::KEY_L;
    case GLFW_KEY_M: return key::KEY_M;
    case GLFW_KEY_N: return key::KEY_N;
    case GLFW_KEY_O: return key::KEY_O;
    case GLFW_KEY_P: return key::KEY_P;
    case GLFW_KEY_Q: return key::KEY_Q;
    case GLFW_KEY_R: return key::KEY_R;
    case GLFW_KEY_S: return key::KEY_S;
    case GLFW_KEY_T: return key::KEY_T;
    case GLFW_KEY_U: return key::KEY_U;
    case GLFW_KEY_V: return key::KEY_V;
    case GLFW_KEY_W: return key::KEY_W;
    case GLFW_KEY_X: return key::KEY_X;
    case GLFW_KEY_Y: return key::KEY_Y;
    case GLFW_KEY_Z: return key::KEY_Z;
    case GLFW_KEY_BACKSPACE: return key::KEY_BACKSPACE;
    case GLFW_KEY_INSERT: return key::KEY_INSERT;
    case GLFW_KEY_HOME: return key::KEY_HOME;
    case GLFW_KEY_PAGE_UP: return key::KEY_PAGE_UP;
    case GLFW_KEY_PAGE_DOWN: return key::KEY_PAGE_DOWN;
    case GLFW_KEY_DELETE: return key::KEY_DELETE;
    case GLFW_KEY_END: return key::KEY_END;
    case GLFW_KEY_TAB: return key::KEY_TAB;
    case GLFW_KEY_ENTER: return key::KEY_ENTER;
    case GLFW_KEY_KP_ENTER: return key::KEY_KP_ENTER;
    case GLFW_KEY_SPACE: return key::KEY_SPACE;
    case GLFW_KEY_COMMA: return key::KEY_COMMA;
    case GLFW_KEY_PERIOD: return key::KEY_DECIMALPOINT;
    case GLFW_KEY_MINUS: return key::KEY_MINUS;
    case GLFW_KEY_LEFT: return key::KEY_LEFT;
    case GLFW_KEY_UP: return key::KEY_UP;
    case GLFW_KEY_RIGHT: return key::KEY_RIGHT;
    case GLFW_KEY_DOWN: return key::KEY_DOWN;
    case GLFW_KEY_ESCAPE: return key::KEY_ESCAPE;
    case GLFW_KEY_RIGHT_ALT: return key::KEY_ALT_R;
    case GLFW_KEY_LEFT_ALT: return key::KEY_ALT_L;
    case GLFW_KEY_RIGHT_SHIFT: return key::KEY_SHIFT_R;
    case GLFW_KEY_LEFT_SHIFT: return key::KEY_SHIFT_L;
    case GLFW_KEY_RIGHT_CONTROL: return key::KEY_CTRL_R;
    case GLFW_KEY_LEFT_CONTROL: return key::KEY_CTRL_L;
    default: return key::KEY_UNKNOWN;
  }
}
//==============================================================================
}  // namespace tatooine::gl
//==============================================================================
#endif
