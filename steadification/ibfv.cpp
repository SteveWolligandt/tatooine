#include "GL/glut.h"
#include <stdlib.h>
#include <math.h>
#include <thread>
#include <array>
#include <string>

//==============================================================================
namespace tatooine::steadification::ibfv {
//==============================================================================

using vec2 = std::array<float, 2>;

float stepsize  = 0.05;
float t         = 0;
float desired_t = 11;

constexpr unsigned int npn   = 512;
constexpr float        scale = 2.0;

int          iframe = 0; 
unsigned int npat   = 32;
int          alpha  = 0.12 * 255;
unsigned int cnt    = 0;
bool         stop   = false;

//==============================================================================
struct dg {
  static constexpr std::array domain_x {0.0f, 2.0f};
  static constexpr std::array domain_y {0.0f, 1.0f};

  static constexpr std::array domain_size {domain_x[1] - domain_x[0],
                                           domain_y[1] - domain_y[0]};
  static constexpr GLsizei width        = 2048;
  static constexpr GLsizei height       = 1024;
  static constexpr unsigned int nmesh_x = 200;
  static constexpr unsigned int nmesh_y = 100;

  //------------------------------------------------------------------------------
  static auto getDP(const vec2& x, float t) {
    static const float pi      = M_PI;
    static const float epsilon = 0.25;
    static const float omega   = 2 * pi * 0.1;
    static const float A       = 0.1;
    float a  = epsilon * sin(omega * t);
    float b  = 1.0 - 2.0 * a;
    float f  = a * x[0] * x[0] + b * x[0];
    float df = 2 * a * x[0] + b;

    return vec2{x[0] - pi * A * sin(pi * f) * cos(pi * x[1])      * stepsize,
                x[1] + pi * A * cos(pi * f) * sin(pi * x[1]) * df * stepsize};
  }
};

//==============================================================================
struct sc {
  static constexpr std::array domain_x {0.0f, 1.0f};
  static constexpr std::array domain_y {0.0f, 1.0f};

  static constexpr std::array domain_size {domain_x[1] - domain_x[0],
                                           domain_y[1] - domain_y[0]};
  static constexpr GLsizei width  = 1024;
  static constexpr GLsizei height = 1024;
  static constexpr unsigned int nmesh_x = 100;
  static constexpr unsigned int nmesh_y = 100;

  //------------------------------------------------------------------------------
  static auto getDP(const vec2& x, float t) {
    return vec2{x[0] - sin(t) * stepsize,
                x[1] + cos(t) * stepsize};
  }
};

//==============================================================================
template <typename vf_t>
void init_gl() { 
  glViewport(0, 0, vf_t::width, vf_t::height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity(); 
  glTranslatef(-1 + vf_t::domain_x[0], -1 + vf_t::domain_x[0], 0);
  glScalef(2.0f/vf_t::domain_size[0], 2.0f / vf_t::domain_size[1], 1);
  glTexParameteri(GL_TEXTURE_2D, 
                  GL_TEXTURE_WRAP_S, GL_REPEAT); 
  glTexParameteri(GL_TEXTURE_2D, 
                  GL_TEXTURE_WRAP_T, GL_REPEAT); 
  glTexParameteri(GL_TEXTURE_2D, 
                  GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, 
                  GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexEnvf(GL_TEXTURE_ENV, 
                  GL_TEXTURE_ENV_MODE, GL_REPLACE);
  glEnable(GL_TEXTURE_2D);
  glShadeModel(GL_FLAT);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glClear(GL_COLOR_BUFFER_BIT);
}

//------------------------------------------------------------------------------
void make_patterns() { 
  int lut[256];
  int phase[npn][npn];
  GLubyte pat[npn][npn][4];
  unsigned int i, j, k, t;
   
  for (i = 0; i < 256; i++)
    lut[i] = i < 127 ? 0 : 255;

  for (i = 0; i < npn; i++)
    for (j = 0; j < npn; j++)
      phase[i][j] = rand() % 256; 

  for (k = 0; k < npat; k++) {
     t = k*256/npat;
     for (i = 0; i < npn; i++) 
       for (j = 0; j < npn; j++) {
         pat[i][j][0] =
         pat[i][j][1] =
         pat[i][j][2] = lut[(t + phase[i][j]) % 255];
         pat[i][j][3] = alpha;
       }
     glNewList(k + 1, GL_COMPILE);
     glTexImage2D(GL_TEXTURE_2D, 0, 4, npn, npn, 0, 
                  GL_RGBA, GL_UNSIGNED_BYTE, pat);
     glEndList();
  }
}

//------------------------------------------------------------------------------
template <typename vf_t>
void display() { 
  float  tmax_x = vf_t::width  / (scale * npn);
  float  tmax_y = vf_t::height / (scale * npn);
  constexpr float  dm_x          = vf_t::domain_size[0] / (vf_t::nmesh_x - 1.0f);
  constexpr float  dm_y          = vf_t::domain_size[1] / (vf_t::nmesh_y - 1.0f);
  unsigned int   i, j;
  float x1, x2, y;


  if (!stop) {
    for (i = 0; i < vf_t::nmesh_x - 1; i++) {
       x1 = dm_x * i;
       x2 = x1 + dm_x;
       glBegin(GL_QUAD_STRIP);
       for (j = 0; j < vf_t::nmesh_y; j++) {
           y = dm_y * j;
           auto dp0 = vf_t::getDP({x1, y}, t);
           glTexCoord2f(x1/vf_t::domain_size[0] - vf_t::domain_x[0], 
                        y/vf_t::domain_size[1] - vf_t::domain_y[0]);
           glVertex2f(dp0[0], dp0[1]);

           auto dp1 = vf_t::getDP({x2, y}, t);
           glTexCoord2f(x2/vf_t::domain_size[0] - vf_t::domain_x[0], 
                        y/vf_t::domain_size[1] - vf_t::domain_y[0]);
           glVertex2f(dp1[0], dp1[1]);
       }
       glEnd();
    }
    ++iframe;

    glEnable(GL_BLEND);
    glCallList(iframe % npat + 1);
    glBegin(GL_QUAD_STRIP);
       glTexCoord2f(0.0,  0.0);      glVertex2f(vf_t::domain_x[0], vf_t::domain_y[0]);
       glTexCoord2f(tmax_x, 0.0);    glVertex2f(vf_t::domain_x[1], vf_t::domain_y[0]);
       glTexCoord2f(0.0,  tmax_y);   glVertex2f(vf_t::domain_x[0], vf_t::domain_y[1]);
       glTexCoord2f(tmax_x, tmax_y); glVertex2f(vf_t::domain_x[1], vf_t::domain_y[1]);
    glEnd();
    glDisable(GL_BLEND);
    glCopyTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                     0, 0, vf_t::width, vf_t::height, 0);
    glutSwapBuffers();
    if (t > desired_t) stop = true;
     t += stepsize;
  }
}

//------------------------------------------------------------------------------
template <typename vf_t>
void calc(int argc, char** argv) {
  t         = atof(argv[2]);
  desired_t = atof(argv[3]);
  stepsize  = atof(argv[4]);

  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB); 
  glutInitWindowSize(vf_t::width, vf_t::height);
  glutCreateWindow(argv[0]);
  glutDisplayFunc(display<vf_t>);
  glutIdleFunc(display<vf_t>);
  init_gl<vf_t>();
  make_patterns();
  // std::this_thread::sleep_for(std::chrono::seconds(5));
  glutMainLoop();
}
//==============================================================================
}  // namespace tatooine::steadification::ibfv
//==============================================================================

//------------------------------------------------------------------------------
int main(int argc, char** argv) { 
  using namespace tatooine::steadification::ibfv;
       if (std::string(argv[1]) == "dg") calc<dg>    (argc, argv);
  else if (std::string(argv[1]) == "sc") calc<sc>(argc, argv);
}
