#include "cuda.h"
#include "galaxy.h"
#include <iostream>

__global__ void kernel_UpdateParticlesPosition(int n , struct point *Shuffled_galaxysParticles, struct  vec3 *ParticulesPositions){
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	

	if ( i < n ){
		Shuffled_galaxysParticles[i].position_x += Shuffled_galaxysParticles[i].velocity_x * 0.1;
		ParticulesPositions[i].x = Shuffled_galaxysParticles[i].position_x;
		Shuffled_galaxysParticles[i].position_y += Shuffled_galaxysParticles[i].velocity_y * 0.1;
		ParticulesPositions[i].y = Shuffled_galaxysParticles[i].position_y;
		Shuffled_galaxysParticles[i].position_z += Shuffled_galaxysParticles[i].velocity_z * 0.1;
		ParticulesPositions[i].z = Shuffled_galaxysParticles[i].position_z;
		
	}
	
}

__global__ void kernel_Update_GPU(int n, int zeta, int M, struct point * Shuffled_galaxysParticles){

	__shared__ float sharedPx[256];
	__shared__ float sharedPy[256];
	__shared__ float sharedPz[256];


	float VP_particle_x = 0.0f;
	float VP_particle_y = 0.0f;
	float VP_particle_z = 0.0f;


	float slope_x = 0.0f;
	float slope_y = 0.0f;
	float slope_z = 0.0f;
	float temp = 0.0f;

	float accx = 0.0f;
	float accy = 0.0f;
	float accz = 0.0f;

	
	
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int iterator = x * 16 + y; 

	VP_particle_x = Shuffled_galaxysParticles[iterator].position_x;
	VP_particle_y = Shuffled_galaxysParticles[iterator].position_y;
	VP_particle_z = Shuffled_galaxysParticles[iterator].position_z;
	

	for (int p = 0 ; p < 16; p++){	
		
		
		int i = threadIdx.y * blockDim.x + threadIdx.x;
		
		sharedPx[i] = Shuffled_galaxysParticles[i +p*256].position_x;
		sharedPy[i] = Shuffled_galaxysParticles[i +p*256].position_y;
		sharedPz[i] = Shuffled_galaxysParticles[i +p*256].position_z;


		__syncthreads();
	

		for(int j =0; j< 256 ; j++){
			if (j != i){


				slope_x = sharedPx[j] - VP_particle_x ;
				slope_y = sharedPy[j] - VP_particle_y ;
				slope_z = sharedPz[j] - VP_particle_z ;



				float d = sqrtf( slope_x * slope_x + slope_y * slope_y + slope_z * slope_z);
				if (d <1.0f)  d = 1.0f ;
				//if (d > 30.0)  d = 30.0 ;

				temp = M * zeta * (1/(d*d*d)) * Shuffled_galaxysParticles[j + p*256].masse ;



				accx += temp*(slope_x);
				accy += temp*(slope_y);
				accz += temp*(slope_z);		
			}

		}

		__syncthreads();


		Shuffled_galaxysParticles[iterator].velocity_x += accx;
		Shuffled_galaxysParticles[iterator].velocity_y += accy;
		Shuffled_galaxysParticles[iterator].velocity_z += accz;
								
		} 
	

}

void UpdateParticlesPosition(int nblocks, int nthreads,int n , struct point *Shuffled_galaxysParticles, struct vec3 *ParticulesPositions){
	kernel_UpdateParticlesPosition<<<nblocks, nthreads>>>( n, Shuffled_galaxysParticles, ParticulesPositions);
}

void Update_GPU(int nblocks, int nthreads, int n , int zeta, int M, struct point * Shuffled_galaxysParticles){
	dim3 grid(16, 1, 1);
	dim3 block(16, 16, 1);
	kernel_Update_GPU<<<grid, block>>>( n , zeta, M, Shuffled_galaxysParticles);
}


