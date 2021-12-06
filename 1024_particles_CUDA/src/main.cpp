#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <iostream>

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
#include "galaxy.h"




static float g_inertia = 0.5f;
static float M = 10;   //Mass Factor
static float zeta = 1; // particles damping factor 

static float oldCamPos[] = { 0.0f, 0.0f, -45.0f };
static float oldCamRot[] = { 0.0f, 0.0f, 0.0f };
static float newCamPos[] = { 0.0f, 0.0f, -45.0f };
static float newCamRot[] = { 0.0f, 0.0f, 0.0f };

static bool g_showGrid = true;
static bool g_showAxes = true;
static bool g_showPart = true;


using namespace std;

static struct point Shuffled_galaxysParticles[MIN_PARTICLES];

void loadonepoint(struct point* p , char * line){
	sscanf(line, "%f %f %f %f %f %f %f" , &(p->masse) ,  &(p->position_x) ,&(p->position_y) , &(p->position_z) , &(p->velocity_x) , &(p->velocity_y) ,
		 &(p->velocity_z));
}

void loadGalaxysParticlesData(char* Filename){
	FILE *inFile;
	char line [100];
	int i = 0;
	int j= 0;

	
	inFile = fopen(Filename, "r");
	
	if(inFile == NULL){
		printf("ERROR::can't open particles data file!!\n"); 
	}
	else{
	
		while((fgets(line, sizeof(line), inFile) != NULL) && (i<MAX_PARTICLES)){	
			//loadonepoint(&galaxysParticles[i] , line);
			//printf("%f\n",galaxysParticles[i].position_y );
			i++;
			if(i%80 == 0)
			{
				loadonepoint(&Shuffled_galaxysParticles[j] , line);
				j++;
			}
				
				
		}

	}
	fclose(inFile);

}

void ShowMilkeyGalaxy(struct vec3 *ParticulesPositions){
	int i ,j ;
	glBegin( GL_POINTS );
	
	for (i = 0; i< 205; i++){
		glColor3f( 1.0f, 0.0f, 0.0f );
		glVertex3f(ParticulesPositions[i].x , ParticulesPositions[i].y, ParticulesPositions[i].z);
	
	}

	for (i = 204; i< 407; i++){
		glColor3f( 0.5625f, 0.92f, 0.5625f );
		glVertex3f(ParticulesPositions[i].x , ParticulesPositions[i].y, ParticulesPositions[i].z);
		
	}


	for (i = 406; i< 513; i++){
		glColor3f( 1.0f, 0.0f, 0.0f );
		glVertex3f(ParticulesPositions[i].x , ParticulesPositions[i].y, ParticulesPositions[i].z);

	}


	for (i = 512; i< 617; i++){

		glColor3f( 0.5625f, 0.92f, 0.5625f );
		glVertex3f(ParticulesPositions[i].x , ParticulesPositions[i].y, ParticulesPositions[i].z);

	}


	for (i = 616; i< 821; i++){
		glColor3f( 1.0f, 0.0f, 0.0f );
		glVertex3f(ParticulesPositions[i].x , ParticulesPositions[i].y, ParticulesPositions[i].z);

	}

	for (i = 820; i< 1024; i++){
		glColor3f( 0.5625f, 0.92f, 0.5625f );
		glVertex3f(ParticulesPositions[i].x , ParticulesPositions[i].y, ParticulesPositions[i].z);
		
	}


	glEnd();
}




void DrawGridXZ( float ox, float oy, float oz, int w, int h, float sz ) {
	
	int i;

	glLineWidth( 1.0f );

	glBegin( GL_LINES );

	glColor3f( 0.48f, 0.48f, 0.48f );

	for ( i = 0; i <= h; ++i ) {
		glVertex3f( ox, oy, oz + i * sz );
		glVertex3f( ox + w * sz, oy, oz + i * sz );
	}

	for ( i = 0; i <= h; ++i ) {
		glVertex3f( ox + i * sz, oy, oz );
		glVertex3f( ox + i * sz, oy, oz + h * sz );
	}

	glEnd();

}

void ShowAxes() {

	glLineWidth( 2.0f );

	glBegin( GL_LINES );
	
	glColor3f( 1.0f, 0.0f, 0.0f );
	glVertex3f( 0.0f, 0.0f, 0.0f );
	glVertex3f( 2.0f, 0.0f, 0.0f );

	glColor3f( 0.0f, 1.0f, 0.0f );
	glVertex3f( 0.0f, 0.0f, 0.0f );
	glVertex3f( 0.0f, 2.0f, 0.0f );

	glColor3f( 0.0f, 0.0f, 1.0f );
	glVertex3f( 0.0f, 0.0f, 0.0f );
	glVertex3f( 0.0f, 0.0f, 2.0f );
	
	glEnd();

}

void ShowOnePart() {

	//glLineWidth( 2.0f );

	glBegin( GL_POINTS );
	glColor3f( 1.0f, 0.0f, 0.0f );
	glVertex3f( 2.0f, 0.0f, 0.0f );
	glEnd();

}

//#define VERBOSE

inline bool CUDA_MALLOC( void ** devPtr, size_t size ) {
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc( devPtr, size );
	if ( cudaStatus != cudaSuccess ) {
		printf( "error: unable to allocate buffer\n");
		return false;
	}
	return true;
}

inline bool CUDA_MEMCPY( void * dst, const void * src, size_t count, enum cudaMemcpyKind kind ) {
	cudaError_t cudaStatus;
	//cudaStream_t streamResult;
	//cudaError_t cudaResult;
	//cudaResult = cudaStreamCreate(&streamResult);
	cudaStatus = cudaMemcpy( dst, src, count, kind);
	if ( cudaStatus != cudaSuccess ) {
		printf( "error: unable to copy buffer\n");
		return false;
	}
	//cudaResult = cudaStreamDestroy(streamResult);
	return true;
}
	
void RandomizeFloatArray( int n, float * arr ) {
	for ( int i = 0; i < n; i++ ) {
		arr[i] = (float)rand() / ( (float)RAND_MAX / 2.0f ) - 1.0f;
	}
}


int main( int argc, char ** argv ) {


	//int device;
	//cudaGetDevice(&device);

	//struct cudaDeviceProp props;
	//cudaGetDeviceProperties(&props, device);
	//printf("Shared memory (Kbytes) %f\n" , (float)(props.sharedMemPerBlock/1024.0));

	//std::cout << hostMinVecs[10].z<< std::endl; 
	//char cwd[256];
	//getcwd(cwd, sizeof(cwd));
	//printf("CUrrent :%s \n", cwd);
	loadGalaxysParticlesData("dubinski.tab");

	SDL_Event event;
	SDL_Window * window;
	SDL_DisplayMode current;
  	
	int width = 640;
	int height = 480;

	bool done = false;

	float mouseOriginX = 0.0f;
	float mouseOriginY = 0.0f;

	float mouseMoveX = 0.0f;
	float mouseMoveY = 0.0f;

	float mouseDeltaX = 0.0f;
	float mouseDeltaY = 0.0f;
	
	if ( argc != 3 ) {
		printf( "usage: cuda vector_size num_threads\n" );
		return 0;
	}
	
	int n = atoi( argv[1] );
	int numThreads = atoi( argv[2] );



	//struct point * deviceMaxPoints = NULL;
	struct point * deviceMinPoints = NULL;
	struct vec3 * deviceMinVecs = NULL;
	struct vec3 * hostMinVecs = NULL;

	

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice( 0 );
	if ( cudaStatus != cudaSuccess ) {
		printf( "error: unable to setup cuda device\n"); exit(0);
	}
	
	

	if (cudaStatus != cudaSuccess) {
		printf( "(EE) Unable to reset device\n" );
	}
	

	hostMinVecs = (struct vec3 *)malloc( sizeof( struct vec3 ) * n );

	CUDA_MALLOC( (void**)&deviceMinPoints,   n * sizeof( struct point ) );
	CUDA_MALLOC( (void**)&deviceMinVecs,   n * sizeof( struct vec3 ) );

	CUDA_MEMCPY( deviceMinPoints, Shuffled_galaxysParticles, n * sizeof( struct point ), cudaMemcpyHostToDevice );
	


	struct timeval begin, end;
	float fps = 0.0f;
	float fpsMin = 10000.0f;
	float fpsMax = -1.0f; 
	float fpsMoyen = 0.0f;
	float totalFps = 0.0f;
	float fpsCount = 0.0f;

	char sfps[44] = "FPS min: ";
	char sfpsMin[44] = "FPS max: ";
	char sfpsMax[44] = "FPS max: ";
	char sfpsMoyen[44] = "FPS moyen: ";

	if ( SDL_Init ( SDL_INIT_EVERYTHING ) < 0 ) {
		printf( "error: unable to init sdl\n" );
		return -1;
	}

	if ( SDL_GetDesktopDisplayMode( 0, &current ) ) {
		printf( "error: unable to get current display mode\n" );
		return -1;
	}

	window = SDL_CreateWindow( "SDL", 	SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 
										width, height, 
										SDL_WINDOW_OPENGL );
  
	SDL_GLContext glWindow = SDL_GL_CreateContext( window );

	GLenum status = glewInit();

	if ( status != GLEW_OK ) {
		printf( "error: unable to init glew\n" );
		return -1;
	}

	if ( ! InitTextRes( "./bin/DroidSans.ttf" ) ) {
		printf( "error: unable to init text resources\n" );
		return -1;
	}

	SDL_GL_SetSwapInterval( 1 );

	while ( !done ) {
  		
		int i;
		//printf("pos x :%f
		while ( SDL_PollEvent( &event ) ) {
      
			unsigned int e = event.type;
			
			if ( e == SDL_MOUSEMOTION ) {
				mouseMoveX = event.motion.x;
				mouseMoveY = height - event.motion.y - 1;
			} else if ( e == SDL_KEYDOWN ) {
				if ( event.key.keysym.sym == SDLK_F1 ) {
					g_showGrid = !g_showGrid;
				} else if ( event.key.keysym.sym == SDLK_F2 ) {
					g_showAxes = !g_showAxes;
				} else if (event.key.keysym.sym == SDLK_F3){
					g_showPart = !g_showPart;

				} else if ( event.key.keysym.sym == SDLK_ESCAPE ) {
 					done = true;
				}
			}

			if ( e == SDL_QUIT ) {
				printf( "quit\n" );
				done = true;
			}

		}

		mouseDeltaX = mouseMoveX - mouseOriginX;
		mouseDeltaY = mouseMoveY - mouseOriginY;

		if ( SDL_GetMouseState( 0, 0 ) & SDL_BUTTON_LMASK ) {
			oldCamRot[ 0 ] += -mouseDeltaY / 5.0f;
			oldCamRot[ 1 ] += mouseDeltaX / 5.0f;
		}else if ( SDL_GetMouseState( 0, 0 ) & SDL_BUTTON_RMASK ) {
			oldCamPos[ 2 ] += ( mouseDeltaY / 100.0f ) * 0.5 * fabs( oldCamPos[ 2 ] );
			oldCamPos[ 2 ]  = oldCamPos[ 2 ] > -5.0f ? -5.0f : oldCamPos[ 2 ];
		}

		mouseOriginX = mouseMoveX;
		mouseOriginY = mouseMoveY;

		glViewport( 0, 0, width, height );
		glClearColor( 0.2f, 0.2f, 0.2f, 1.0f );
		glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
		glEnable( GL_BLEND );
		glBlendEquation( GL_FUNC_ADD );
		glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
		glDisable( GL_TEXTURE_2D );
		glEnable( GL_DEPTH_TEST );
		glMatrixMode( GL_PROJECTION );
		glLoadIdentity();
		gluPerspective( 50.0f, (float)width / (float)height, 0.1f, 100000.0f );
		glMatrixMode( GL_MODELVIEW );
		glLoadIdentity();

		for ( i = 0; i < 3; ++i ) {
			newCamPos[ i ] += ( oldCamPos[ i ] - newCamPos[ i ] ) * g_inertia;
			newCamRot[ i ] += ( oldCamRot[ i ] - newCamRot[ i ] ) * g_inertia;
		}

		glTranslatef( newCamPos[0], newCamPos[1], newCamPos[2] );
		glRotatef( newCamRot[0], 1.0f, 0.0f, 0.0f );
		glRotatef( newCamRot[1], 0.0f, 1.0f, 0.0f );
		
		if ( g_showGrid ) {
			DrawGridXZ( -100.0f, 0.0f, -100.0f, 20, 20, 10.0 );
		}

		if ( g_showAxes ) {
			ShowAxes();
		}


		gettimeofday( &begin, NULL );

		if ( g_showPart ) {
			int numBlocks = ( n + ( numThreads - 1 ) ) / numThreads;

			Update_GPU(numBlocks, numThreads, n , zeta, M, deviceMinPoints);
			UpdateParticlesPosition(numBlocks, numThreads, n, deviceMinPoints, deviceMinVecs );

			cudaStatus = cudaDeviceSynchronize();
	
			if ( cudaStatus != cudaSuccess ) {
				printf( "error: unable to synchronize threads\n");
			}

			CUDA_MEMCPY(hostMinVecs  , deviceMinVecs, n * sizeof( struct vec3 ), cudaMemcpyDeviceToHost );
			
			
			gettimeofday( &end, NULL );
			
			ShowMilkeyGalaxy(hostMinVecs);
		}
		// Simulation should be computed here

		

		fps = (float)1.0f / ( ( end.tv_sec - begin.tv_sec ) * 1000000.0f + end.tv_usec - begin.tv_usec) * 1000000.0f;

		if (fps<fpsMin){
			fpsMin = fps;
		}

		if (fps>fpsMax){
			fpsMax = fps;
		}
		
		totalFps += fps;
		fpsCount++;	
		
		fpsMoyen = totalFps/fpsCount;

		sprintf( sfps, "FPS inst: %.4f", fps );
		sprintf( sfpsMin, "FPS min: %.4f", fpsMin );
		sprintf( sfpsMax, "FPS max: %.4f", fpsMax );
		sprintf( sfpsMoyen, "FPS moyen: %.4f", fpsMoyen );

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluOrtho2D(0, width, 0, height);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		DrawText( 10, height - 20, sfps, TEXT_ALIGN_LEFT, RGBA(255, 255, 255, 255) );
		DrawText( 10, height - 40, sfpsMin, TEXT_ALIGN_LEFT, RGBA(255, 255, 255, 255) );
		DrawText( 10, height - 60, sfpsMax, TEXT_ALIGN_LEFT, RGBA(255, 255, 255, 255) );
		DrawText( 10, height - 80, sfpsMoyen, TEXT_ALIGN_LEFT, RGBA(255, 255, 255, 255) );
		DrawText( 10, 30, "'F1' : show/hide grid", TEXT_ALIGN_LEFT, RGBA(255, 255, 255, 255) );
		DrawText( 10, 10, "'F2' : show/hide axes", TEXT_ALIGN_LEFT, RGBA(255, 255, 255, 255) );

		SDL_GL_SwapWindow( window );
		SDL_UpdateWindowSurface( window );
	}


	SDL_GL_DeleteContext( glWindow );
	DestroyTextRes();
	SDL_DestroyWindow( window );
	cudaStatus = cudaDeviceReset();
	SDL_Quit();

	return 1;
}
