#include "cuda.h"
#include "galaxy.h"


__global__ void kernel_saxpy( int n, float a, float * x, float * y, float * z ) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i < n ) { 
		z[i] = a * x[i] + y [i];
	}
}

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

int i = blockIdx.x * blockDim.x + threadIdx.x;

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


	if(i<n){

		VP_particle_x = Shuffled_galaxysParticles[i].position_x;
		VP_particle_y = Shuffled_galaxysParticles[i].position_y;
		VP_particle_z = Shuffled_galaxysParticles[i].position_z;

		for(int j =0; j< n ; j++){
			if (j != i){

				
				slope_x = Shuffled_galaxysParticles[j].position_x - VP_particle_x;
				slope_y = Shuffled_galaxysParticles[j].position_y - VP_particle_y;
				slope_z = Shuffled_galaxysParticles[j].position_z - VP_particle_z;


				float d = sqrtf( slope_x * slope_x + slope_y * slope_y + slope_z * slope_z);
				if (d <1.0f)  d = 1.0f ;
				//if (d > 30.0)  d = 30.0 ;

				temp = M * zeta * (1/(d*d*d)) * Shuffled_galaxysParticles[j].masse ;
				accx += temp*(slope_x);
				accy += temp*(slope_y);
				accz += temp*(slope_z);		
			}
	
		}
		Shuffled_galaxysParticles[i].velocity_x += accx;
		Shuffled_galaxysParticles[i].velocity_y += accy;
		Shuffled_galaxysParticles[i].velocity_z += accz;		
	}



 	

}

void saxpy( int nblocks, int nthreads, int n, float a, float * x, float * y, float * z ) {
	kernel_saxpy<<<nblocks, nthreads>>>( n, a, x, y, z );
}

void UpdateParticlesPosition(int nblocks, int nthreads,int n , struct point *Shuffled_galaxysParticles, struct vec3 *ParticulesPositions){
	kernel_UpdateParticlesPosition<<<nblocks, nthreads>>>( n, Shuffled_galaxysParticles, ParticulesPositions);
}

void Update_GPU(int nblocks, int nthreads, int n , int zeta, int M, struct point * Shuffled_galaxysParticles){
	kernel_Update_GPU<<<nblocks, nthreads>>>( n , zeta, M, Shuffled_galaxysParticles);
	
}




/*



*/
