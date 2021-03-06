#ifndef GALAXY_H
#define GALAXY_H


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cuda_runtime.h"
#include "kernel.cuh"

#include <stdbool.h>
#include <math.h>
#include <sys/time.h>
#include <unistd.h>

#include <omp.h>

#include "GL/glew.h"

#include "SDL2/SDL.h"
#include "SDL2/SDL_opengl.h"

#include "text.h"



#define MAX_PARTICLES 81920
#define MIN_PARTICLES 4096



struct point{
	float masse;
	float position_x;
	float position_y;
	float position_z;
	float velocity_x;
	float velocity_y;
	float velocity_z;
};

struct vec3{
	float x;
	float y;
	float z;
};

#endif
