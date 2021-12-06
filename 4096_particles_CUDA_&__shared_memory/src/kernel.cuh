#ifndef __KERNEL_CUH__
#define __KERNEL_CUH__
#include "galaxy.h"


void saxpy( int ngrid, int nblock, int n, float a, struct vec3 hostMinVecs );

//void addAcceleration(int nblocks, int nthreads, int n,struct point *particle,struct point neighbor);

void UpdateParticlesPosition(int nblocks, int nthreads,int n , struct point *Shuffled_galaxysParticles, struct vec3 *ParticulesPositions);

void Update_GPU(int nblocks, int nthreads, int n , int zeta, int M, struct point * Shuffled_galaxysParticles);

#endif


	
