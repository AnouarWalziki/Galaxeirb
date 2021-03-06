#ifndef GALAXY_H
#define GALAXY_H

#include <stdio.h>

#define MIN_PARTICLES 4096
#define MAX_PARTICLES 81920

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
