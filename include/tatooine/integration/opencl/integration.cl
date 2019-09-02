#ifndef __TATOOINE_INTEGRATION_CL__
#define __TATOOINE_INTEGRATION_CL__

float integrate_1f(
  float(f)(float x, float t), 
  float(integrator)(float(f)(float x, float t, float step_width), float x, float t, float step_width), 
  float x0, float t0, float tau, float step_width) {

  float integrated_time = 0;
  float t = t0;
  float x = x0;

  while (integrated_time < tau) {
    x = integrator(f, x, t, step_width);
    t += step_width;
    integrated_time += step_width;
    if (t > t0 + tau) {
      t = t0 + tau;
      integrated_time = tau;
    }
  }

  return x;
}

float2 integrate_2f(
  float2(f)(float2 x, float t), 
  float2(integrator)(float2(f)(float2 x, float t), float2 x, float t, float step_width), 
  float2 x0, float t0, float tau, float step_width) {

  float integrated_time = 0;
  float t = t0;
  float2 x = x0;

  while (integrated_time < tau) {
    x = integrator(f, x, t, step_width);
    t += step_width;
    integrated_time += step_width;
    if (t > t0 + tau) {
      t = t0 + tau;
      integrated_time = tau;
    }
  }

  return x;
}

float3 integrate_3f(
  float3(f)(float3 x, float t), 
  float3(integrator)(float3(f)(float3 x, float t, float step_width), float3 x, float t, float step_width), 
  float3 x0, float t0, float tau, float step_width) {

  float integrated_time = 0;
  float t = t0;
  float3 x = x0;

  while (integrated_time < tau) {
    x = integrator(f, x, t, step_width);
    t += step_width;
    integrated_time += step_width;
    if (t > t0 + tau) {
      t = t0 + tau;
      integrated_time = tau;
    }
  }

  return x;
}

float4 integrate_4f(
  float4(f)(float4 x, float t), 
  float4(integrator)(float4(f)(float4 x, float t, float step_width), float4 x, float t, float step_width), 
  float4 x0, float t0, float tau, float step_width) {

  float integrated_time = 0;
  float t = t0;
  float4 x = x0;

  while (integrated_time < tau) {
    x = integrator(f, x, t, step_width);
    t += step_width;
    integrated_time += step_width;
    if (t > t0 + tau) {
      t = t0 + tau;
      integrated_time = tau;
    }
  }

  return x;
}


#if defined(cl_khr_fp64) || defined(cl_amd_fp64)
double integrate_1d(
  double(f)(double x, double t), 
  double(integrator)(double(f)(double x, double t, double step_width), double x, double t, double step_width), 
  double x0, double t0, double tau, double step_width) {

  double integrated_time = 0;
  double t = t0;
  double x = x0;

  while (integrated_time < tau) {
    x = integrator(f, x, t, step_width);
    t += step_width;
    integrated_time += step_width;
    if (t > t0 + tau) {
      t = t0 + tau;
      integrated_time = tau;
    }
  }

  return x;
}

double2 integrate_2d(
  double2(f)(double2 x, double t), 
  double2(integrator)(double2(f)(double2 x, double t, double step_width), double2 x, double t, double step_width), 
  double2 x0, double t0, double tau, double step_width) {

  double integrated_time = 0;
  double t = t0;
  double2 x = x0;

  while (integrated_time < tau) {
    x = integrator(f, x, t, step_width);
    t += step_width;
    integrated_time += step_width;
    if (t > t0 + tau) {
      t = t0 + tau;
      integrated_time = tau;
    }
  }

  return x;
}

double3 integrate_3d(
  double3(f)(double3 x, double t), 
  double3(integrator)(double3(f)(double3 x, double t, double step_width), double3 x, double t, double step_width), 
  double3 x0, double t0, double tau, double step_width) {

  double integrated_time = 0;
  double t = t0;
  double3 x = x0;

  while (integrated_time < tau) {
    x = integrator(f, x, t, step_width);
    t += step_width;
    integrated_time += step_width;
    if (t > t0 + tau) {
      t = t0 + tau;
      integrated_time = tau;
    }
  }

  return x;
}

double4 integrate_4d(
  double4(f)(double4 x, double t), 
  double4(integrator)(double4(f)(double4 x, double t, double step_width), double4 x, double t, double step_width), 
  double4 x0, double t0, double tau, double step_width) {

  double integrated_time = 0;
  double t = t0;
  double4 x = x0;

  while (integrated_time < tau) {
    x = integrator(f, x, t, step_width);
    t += step_width;
    integrated_time += step_width;
    if (t > t0 + tau) {
      t = t0 + tau;
      integrated_time = tau;
    }
  }

  return x;
}
#endif
#endif