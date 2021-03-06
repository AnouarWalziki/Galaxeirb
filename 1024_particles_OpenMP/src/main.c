#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>

#include <omp.h>

#include "GL/glew.h"

#include "SDL2/SDL.h"
#include "SDL2/SDL_opengl.h"

#include "text.h"

#define MAX_PARTICLES 81920
#define MIN_PARTICLES 1024


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


static struct point galaxysParticles[MAX_PARTICLES];// will contain particles data
static struct point Shuffled_galaxysParticles[MIN_PARTICLES];
static struct vec3 ParticlesAcceleration[MIN_PARTICLES];


struct vec3 Vec3(float x, float y, float z){
	struct vec3 V;
	V.x = x;
	V.y = y;
	V.z = z;
	return V;

}

struct vec3 add(struct vec3 u, struct vec3 v){
	return Vec3(u.x + v.x , u.y + v.y , u.z + v.z);
}

struct vec3 substract(struct vec3 u, struct vec3 v){
	return Vec3(u.x - v.x , u.y - v.y , u.z - v.z);
}

struct vec3 multiply(struct vec3 u, struct vec3 v){
	return Vec3(u.x * v.x , u.y * v.y , u.z * v.z);
}

struct vec3 dot(struct vec3 u, float a){
	return Vec3(u.x *a , u.y * a , u.z * a);
}




void loadonepoint(struct point* p , char * line){
	sscanf(line, "%f %f %f %f %f %f %f" , &(p->masse) ,  &(p->position_x) ,&(p->position_y) , &(p->position_z) , &(p->velocity_x) , &(p->velocity_y) ,
		 &(p->velocity_z));
}

void loadGalaxysParticlesData(char* Filename){
	FILE *inFile;
	char line [100];
	int i = 0;
	int j = 0;

	
	inFile = fopen(Filename, "r");
	
	if(inFile == NULL){
		printf("ERROR::can't open particles data file!!\n"); 
	}
	else{
	
		while((fgets(line, sizeof(line), inFile) != NULL) && (i<MAX_PARTICLES)){	
			loadonepoint(&galaxysParticles[i] , line);
			//printf("%f\n",galaxysParticles[i].position_y );
			i++;
			if (i%80==0){
				loadonepoint(&Shuffled_galaxysParticles[j] , line);
				j++;
			}	
		}

	}
	fclose(inFile);

}


void addAcceleration(struct point *particle,struct point neighbor){
	
	struct vec3 VP_particle = Vec3(particle->position_x , particle->position_y , particle->position_z);
	struct vec3 VP_neighbor = Vec3(neighbor.position_x , neighbor.position_y , neighbor.position_z);

	struct vec3 slope = substract(VP_neighbor , VP_particle);

	float d = sqrtf( slope.x * slope.x + slope.y * slope.y + slope.z * slope.z);
	if (d <1.0f)  d = 1.0f ;
	//if (d > 30.0)  d = 30.0 ;

	float temp = M * zeta * (1/(d*d*d)) * neighbor.masse ;
	
	struct vec3 acc = dot(slope, temp);
	
	particle->velocity_x += acc.x;
	particle->velocity_y += acc.y;
	particle->velocity_z += acc.z;
}  

void UpdateParticlesPosition(struct point *particle){

	particle->position_x += particle->velocity_x * 0.1;
	particle->position_y += particle->velocity_y * 0.1;
	particle->position_z += particle->velocity_z * 0.1;
}

void update(){
	int i;
	int j;
#pragma omp parallel for
	for(i = 0; i<MIN_PARTICLES; i++){
#pragma omp parallel for 
		for(j =0; j< MIN_PARTICLES ; j++){
			if (j != i){

				addAcceleration(&Shuffled_galaxysParticles[i] , Shuffled_galaxysParticles[j]);
				
			}
		}		
	}
//#pragma omp parallel for
	for(i = 0; i<MIN_PARTICLES; i++){
		UpdateParticlesPosition(&Shuffled_galaxysParticles[i]);	
	}
}



void ShowMilkeyGalaxy(){
	int i = 0;
	glBegin( GL_POINTS );

	for (i = 0; i< 205; i++){
		glColor3f( 1.0f, 0.0f, 0.0f );
		glVertex3f(Shuffled_galaxysParticles[i].position_x , Shuffled_galaxysParticles[i].position_y, Shuffled_galaxysParticles[i].position_z);
	
	}

	for (i = 204; i< 407; i++){
		glColor3f( 0.5625f, 0.92f, 0.5625f );
		glVertex3f(Shuffled_galaxysParticles[i].position_x , Shuffled_galaxysParticles[i].position_y, Shuffled_galaxysParticles[i].position_z);
		
	}


	for (i = 406; i< 513; i++){
		glColor3f( 1.0f, 0.0f, 0.0f );
		glVertex3f(Shuffled_galaxysParticles[i].position_x , Shuffled_galaxysParticles[i].position_y, Shuffled_galaxysParticles[i].position_z);

	}


	for (i = 512; i< 617; i++){

		glColor3f( 0.5625f, 0.92f, 0.5625f );
		glVertex3f(Shuffled_galaxysParticles[i].position_x , Shuffled_galaxysParticles[i].position_y, Shuffled_galaxysParticles[i].position_z);

	}


	for (i = 616; i< 821; i++){
		glColor3f( 1.0f, 0.0f, 0.0f );
		glVertex3f(Shuffled_galaxysParticles[i].position_x , Shuffled_galaxysParticles[i].position_y, Shuffled_galaxysParticles[i].position_z);

	}

	for (i = 820; i< 1024; i++){
		glColor3f( 0.5625f, 0.92f, 0.5625f );
		glVertex3f(Shuffled_galaxysParticles[i].position_x , Shuffled_galaxysParticles[i].position_y, Shuffled_galaxysParticles[i].position_z);
		
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

int main( int argc, char ** argv ) {

	omp_set_num_threads( 4 );

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

	struct timeval begin, end;
	float fps = 0.0;
	char sfps[40] = "FPS: ";

	if ( SDL_Init ( SDL_INIT_EVERYTHING ) < 0 ) {
		printf( "error: unable to init sdl\n" );
		return -1;
	}

	if ( SDL_GetDesktopDisplayMode( 0, &current ) ) {
		printf( "error: unable to get current display mode\n" );
		return -1;
	}

	window = SDL_CreateWindow( "SDL", 	SDL_WINDOWPOS_CENTERED, 
										SDL_WINDOWPOS_CENTERED, 
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
			update();
			ShowMilkeyGalaxy();
		}
		// Simulation should be computed here

		gettimeofday( &end, NULL );

		fps = (float)1.0f / ( ( end.tv_sec - begin.tv_sec ) * 1000000.0f + end.tv_usec - begin.tv_usec) * 1000000.0f;
		sprintf( sfps, "FPS : %.4f", fps );

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluOrtho2D(0, width, 0, height);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		DrawText( 10, height - 20, sfps, TEXT_ALIGN_LEFT, RGBA(255, 255, 255, 255) );
		DrawText( 10, 30, "'F1' : show/hide grid", TEXT_ALIGN_LEFT, RGBA(255, 255, 255, 255) );
		DrawText( 10, 10, "'F2' : show/hide axes", TEXT_ALIGN_LEFT, RGBA(255, 255, 255, 255) );

		SDL_GL_SwapWindow( window );
		SDL_UpdateWindowSurface( window );
	}

	SDL_GL_DeleteContext( glWindow );
	DestroyTextRes();
	SDL_DestroyWindow( window );
	SDL_Quit();

	return 1;
}

