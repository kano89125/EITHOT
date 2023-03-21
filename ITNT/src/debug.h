#include <cstdio>

#ifdef DEBUG
	#define PRINT(s, ...) printf((s), ##__VA_ARGS__)
#else
	#define PRINT(s, ...)
#endif
