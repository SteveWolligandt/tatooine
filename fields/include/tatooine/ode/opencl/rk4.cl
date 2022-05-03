#ifndef __TATOOINE_RK4_CL__
#define __TATOOINE_RK4_CL__

#include "integration.cl"

float rk4_1f(float(f)(float x, float t), float x, float t, float step_width) {
  float k1 = f(x,                          t                    );
  float k2 = f(x + 0.5f * step_width * k1, t + 0.5f * step_width);
  float k3 = f(x + 0.5f * step_width * k2, t + 0.5f * step_width);
  float k4 = f(x +        step_width * k3, t +        step_width);
  return x + step_width * (k1 + 2*k2 + 2*k3 + k4) / 6.0f;
}
float2 rk4_2f(float2(f)(float2 x, float t), float2 x, float t, float step_width) {
  float2 k1 = f(x,                          t                    );
  float2 k2 = f(x + 0.5f * step_width * k1, t + 0.5f * step_width);
  float2 k3 = f(x + 0.5f * step_width * k2, t + 0.5f * step_width);
  float2 k4 = f(x +        step_width * k3, t +        step_width);
  return x + step_width * (k1 + 2*k2 + 2*k3 + k4) / 6.0f;
}
float3 rk4_3f(float3(f)(float3 x, float t), float3 x, float t, float step_width) {
  float3 k1 = f(x,                          t                    );
  float3 k2 = f(x + 0.5f * step_width * k1, t + 0.5f * step_width);
  float3 k3 = f(x + 0.5f * step_width * k2, t + 0.5f * step_width);
  float3 k4 = f(x +        step_width * k3, t +        step_width);
  return x + step_width * (k1 + 2*k2 + 2*k3 + k4) / 6.0f;
}
float4 rk4_4f(float4(f)(float4 x, float t), float4 x, float t, float step_width) {
  float4 k1 = f(x,                          t                    );
  float4 k2 = f(x + 0.5f * step_width * k1, t + 0.5f * step_width);
  float4 k3 = f(x + 0.5f * step_width * k2, t + 0.5f * step_width);
  float4 k4 = f(x +        step_width * k3, t +        step_width);
  return x + step_width * (k1 + 2.0f*k2 + 2.0f*k3 + k4) / 6.0f;
}

float integrate_rk4_1f(float(f)(float x, float t), float x0, float t0, float tau, float step_width) {
  return integrate(f, &rk4_1f, x0, t0, tau, step_width);
}
float2 integrate_rk4_2f(float2(f)(float2 x, float t), float2 x0, float t0, float tau, float step_width) {
  return integrate_2f(f, &rk4_2f, x0, t0, tau, step_width);
}
float3 integrate_rk4_3f(float3(f)(float3 x, float t), float3 x0, float t0, float tau, float step_width) {
  return integrate(f, &rk4_3f, x0, t0, tau, step_width);
}
float4 integrate_rk4_4f(float4(f)(float4 x, float t), float4 x0, float t0, float tau, float step_width) {
  return integrate(f, &rk4_4f, x0, t0, tau, step_width);
}

#if defined(cl_khr_fp64) || defined(cl_amd_fp64)
double rk4_1d(double(f)(double x, double t), double x, double t, double step_width) {
  double k1 = f(x,                          t                    );
  double k2 = f(x + 0.5 * step_width * k1, t + 0.5 * step_width);
  double k3 = f(x + 0.5 * step_width * k2, t + 0.5 * step_width);
  double k4 = f(x +        step_width * k3, t +        step_width);
  return x + step_width * (k1 + 2*k2 + 2*k3 + k4) / 6.0;
}
double2 rk4_2d(double2(f)(double2 x, double t), double2 x, double t, double step_width) {
  double2 k1 = f(x,                          t                    );
  double2 k2 = f(x + 0.5 * step_width * k1, t + 0.5 * step_width);
  double2 k3 = f(x + 0.5 * step_width * k2, t + 0.5 * step_width);
  double2 k4 = f(x +        step_width * k3, t +        step_width);
  return x + step_width * (k1 + 2*k2 + 2*k3 + k4) / 6.0;
}
double3 rk4_3d(double3(f)(double3 x, double t), double3 x, double t, double step_width) {
  double3 k1 = f(x,                          t                    );
  double3 k2 = f(x + 0.5 * step_width * k1, t + 0.5 * step_width);
  double3 k3 = f(x + 0.5 * step_width * k2, t + 0.5 * step_width);
  double3 k4 = f(x +        step_width * k3, t +        step_width);
  return x + step_width * (k1 + 2*k2 + 2*k3 + k4) / 6.0;
}
double4 rk4_4d(double4(f)(double4 x, double t), double4 x, double t, double step_width) {
  double4 k1 = f(x,                          t                    );
  double4 k2 = f(x + 0.5 * step_width * k1, t + 0.5 * step_width);
  double4 k3 = f(x + 0.5 * step_width * k2, t + 0.5 * step_width);
  double4 k4 = f(x +        step_width * k3, t +        step_width);
  return x + step_width * (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0;
}
double integrate_rk4_1d(double(f)(double x, double t), double x0, double t0, double tau, double step_width) {
  return integrate(f, &rk4_1d, x0, t0, tau, step_width);
}
double2 integrate_rk4_2d(double2(f)(double2 x, double t), double2 x0, double t0, double tau, double step_width) {
  return integrate(f, &rk4_2d, x0, t0, tau, step_width);
}
double3 integrate_rk4_3d(double3(f)(double3 x, double t), double3 x0, double t0, double tau, double step_width) {
  return integrate(f, &rk4_3d, x0, t0, tau, step_width);
}
double4 integrate_rk4_4d(double4(f)(double4 x, double t), double4 x0, double t0, double tau, double step_width) {
  return integrate(f, &rk4_4d, x0, t0, tau, step_width);
}
#endif
#endif