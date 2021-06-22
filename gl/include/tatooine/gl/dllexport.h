#ifndef DLL_API
#ifdef __linux__ 
#define DLL_API
#elif _WIN32
#ifdef DLL_EXPORT
#define DLL_API __declspec(dllexport) 
#else
#define DLL_API __declspec(dllimport) 
#endif
#else
#endif
#endif
