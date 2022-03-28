// 02.05.2021 - Colloid in a fluctuating scalar field
// Davide Venturelli & Benjamin Walter
// Last update 08.11.21

/* COMMENTS:
- Stochastic Runge-Kutta II for colloid evolution, Euler-Maruyama for field evolution (can be enhanced, but the price is O(N) at least. We could even think of anisotropic resolution, i.e. better around the colloid).
- No boundary conditions on the colloid displacement; they only get enforced when locating the nearest site.
- Space is measured in units of the lattice spacing.
- I am saving X(t) in colloid_pos and Y(t) in colloid_msd. Printing only X(t) at the moment.
- Finite size colloid, Gaussian interaction potential.
- Zero set in correspondence of trap center when printing out colloid data.
- Random inizialitazion of the field in Fourier space
- There is no limit on the size of R (other than physical, L/2)
- Compile as $ gcc -o colloid colloid_RF.c -lm -lgsl -lfftw3 -O3 -Wall -Wextra -O3
*/

// LIBRARIES, TYPES, DEFINITIONS

#include <stdio.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf.h>
#include <time.h>
#include <unistd.h>
#include <fftw3.h>

#define ALLOC(p,n) (p)=malloc( (n) * sizeof(*(p))); if( (p) == NULL){printf("Allocation of '%s' failed. Terminate. \n", #p); exit(2); } 
#define CALLOC(p,n) (p)=calloc( (n) , sizeof(*(p))); if( (p) == NULL){printf("Allocation of '%s' failed. Terminate. \n", #p); exit(2); } 
//#define long unsigned long						 					// This is to make long hold twice as much
#define DD printf("# Debug: line %d \n",__LINE__);
#define DEBUG (1)
#define EPS 0.0000000001												// Used to compare non-integer variables to 0

// Data types
typedef struct{
	double mass;														// Mass of the field
	double lambda;														// Field-colloid coupling strength
	double quartic_u;													// Self-interaction coupling strength
	double temperature;													// Temperature of the common bath
	double relativeD;													// Ratio of colloid to field mobility (it's basically \nu of the particle)
	double trap_strength;												// Stiffness of the harmonic trap
	int rng_seed;														// Seed for random number generator
	int system_size;													// Side of the DIM-dimensional lattice
	double delta_t;														// Time-discretization step
	long n_timestep;													// Number of timesteps
	int mc_runs;														// Number of Monte Carlo iterations
	double R;															// Size of colloid in lattice units
	double X0;															// Initial displacement of the colloid
} parameter;

typedef struct{															// This structure bundles all observables 
	double** field_average;												// Saves the measured field-average <phi[i]> for certain subset of i's (eg along an axis) AND at all writing times
	double** field_correlation;											// Saves the measured field-correlator < phi[0] phi[i]> for certain subset of i's (eg along an axis) AND at all writing times
	double* colloid_pos;												// Saves the measured displacement of the colloid at all writing times
	double* colloid_msd;												// Saves the measured mean square displacement of the colloid at all writing times
	double write_time_delta;											// This is the gap between writing times (at the moment linear, maybe later exponential writing times?) 
	int write_count;
//	double* colloid_fpt_distribution;									// Saves the first-passage time distribution of the colloid
} observables;

// Global variables
int DIM = 1;															// Physical dimensions
int MOD = 2;															// 0 for Model A, 2 for Model B
int top;																// Actual number of sites within 5R around a single site
double gauss_norm;														// Normalization constant of Gaussian functions
const gsl_rng_type *T;													// GSL RNG (it turns out Ziggurat is faster than Box-Muller)
gsl_rng *seed;


// FUNCTION PROTOTYPES

void default_parameters(parameter*);
void initialise(long***, parameter*, long***);
void initialise_observable(observables*, parameter*);
void wipe(double**, double**, parameter*);
void prethermalize(double**, parameter*);
void field_prepare(double**, parameter*);
void evolveB(double**, double**, long**, parameter*, observables*, long**);
void evolveA(double**, double**, long**, parameter*, observables*, long**);
void laplacian(double**, double*, long**, long);
void laplacian_of_cube(double**, double*, long**, long);
void generate_noise_field(double**, long, parameter*);
void gradient_field(double**, double*, long**, long);
void phi_evolveB(double**, double*, double*, double*, double*, long, parameter*, double*, long**);
void phi_evolveA(double**, double*, double*, long, parameter*, double*, long**);
void measure(double**, double**, long, parameter*, observables*);
void print_observables(observables*, parameter*);
void print_trajectory(observables*, parameter*);
void print_params(parameter*);
void printhelp(parameter*);
void print_source(void);
void neighborhood(long***, int);
int ind2coord(int, int, int);
int vec2ind(int*, int);
void ind2vecROW(int**, int, int, int);
int closest_site(double*, int);
double distance(double*, int, int);
unsigned modulo(int, unsigned);
double modular(double, double, double);
int factorial(int);
int double_factorial(int);
int intpow(int, int);
double floatpow(double, int);
void print_lookup(long **, int, int);
double gaussian(double, double);
double laplacian_gaussian(double, double);
void evolve_quenched(double**, double**, parameter*, observables*, long**);


// MAIN BODY

int main(int argc, char *argv[]){
	setlinebuf(stdout);
	
	// INPUT PARAMETERS
	parameter params;
	default_parameters(&params);

	opterr = 0;
	int c = 0;
	// ./colloid -r 2 -L 23 (order doesn't count)
    while( (c = getopt (argc, argv, "L:r:l:u:T:d:k:S:t:N:M:n:m:R:X:hP") ) != -1){
    	switch(c){
			case 'L':
				params.system_size = atoi(optarg);
				break;
			case 'r':
				params.mass = atof(optarg);
				break;
			case 'l':
				params.lambda = atof(optarg);
				break;
			case 'u':
				params.quartic_u = atof(optarg);
				break;
			case 'T':
				params.temperature = atof(optarg);
				break;
			case 'n':
				params.relativeD = atof(optarg);
				break;
			case 'k':
				params.trap_strength = atof(optarg);
				break;
			case 'S':
				params.rng_seed = atoi(optarg);
				break;
			case 't':
				params.delta_t = atof(optarg);
				break;
			case 'N':
				params.n_timestep = atol(optarg);
				break;
			case 'M':
				params.mc_runs = atoi(optarg);
				break;
			case 'd':
				DIM = atoi(optarg);
				break;
			case 'm':
				MOD = atoi(optarg);
				break;
			case 'R':
				params.R = atof(optarg);
				break;
			case 'X':
				params.X0 = atof(optarg);
				break;
			case 'P':
				print_source();
				exit(2);
			case 'h':
				printhelp(&params);
				exit(2);
			default:
				printhelp(&params);
                exit(EXIT_FAILURE);
       }
	}
	
	// VARIABLES, ALLOCATION
	int i, j;
	long L = params.system_size;
	long n_sites = intpow(L, DIM);
	int max_within_5R;
	double prefactor[3] = {2,M_PI,4.0/3*M_PI};
	if(DIM<4) max_within_5R = (int) ceil( prefactor[DIM-1]*floatpow(ceil(5*params.R), DIM) ) + 2;
	else max_within_5R=10000;											// Overkill, maybe I'll generalize it one day
	
	double* phi; 														// Field on lattice NxNxN
	double* y_colloid; 													// Colloid position (DIM real numbers)
	long** neighbours; 													// i x j - table with j neighbours of site i
	long** mosaic;														// Table with neighbours of site i with distance < 4R
	
	ALLOC(phi, n_sites);
	CALLOC(y_colloid, DIM);
	ALLOC(neighbours, n_sites);
	ALLOC(mosaic, n_sites);
	
	for(i = 0; i < n_sites; i++){
		neighbours[i]=calloc( 2 * DIM , sizeof(long)); 					// At each index there are 2*D neighbours
		if( neighbours[i] == NULL){printf("Allocation of neighbour list failed. Terminate. \n"); exit(2);}
		mosaic[i]=calloc( max_within_5R , sizeof(long)); 				// At each index, all the sites within 5R
		if( mosaic[i] == NULL){printf("Allocation of mosaic list failed. Terminate. \n"); exit(2);}
	}		

	// INITIALIZATION
	initialise(&neighbours, &params, &mosaic);							// Random function, nearest neighbours list, mosaic
	observables obvs;													// Creates a pointer to an observables structure
	initialise_observable(&obvs, &params);								// Initialise observables
	print_params(&params);   											// Print header with all parameters
	gauss_norm = 1/(pow(2*M_PI,DIM*0.5) * floatpow(params.R,DIM));		// Normalization for Gaussian functions

	printf("# Max within 5R: %d\n", max_within_5R);
	printf("# Actual within 5R: %d\n", top);
	//print_lookup(neighbours, n_sites, 2*DIM);							// Prints nearest-neighbours list
	//print_lookup(mosaic, n_sites, top);								// Prints cell-list

	// MC ITERATION
	int mc_counter, flag;	
	printf("\n# MC ITERATION BEGINS\n");
		
	for(mc_counter = 0; mc_counter < params.mc_runs; mc_counter++){
		//wipe(&phi, &y_colloid, &params); 								// Reset field and colloid to initial conditions
		prethermalize(&phi, &params);									// Pre-thermalization cycle
		//field_prepare(&phi, &params);	
		for(j=0; j<DIM; j++) y_colloid[j] = L/2;						// Colloid initially in the middle of the trap
		y_colloid[0] = (L/2)+params.X0;									// Add initial displacement in one direction				
		
		// Numerical integration of the dynamics

		if(MOD==0){														// Model A
			evolveA(&phi, &y_colloid, neighbours, &params, &obvs, mosaic);
		} else if(MOD==2){												// Model B
			evolveB(&phi, &y_colloid, neighbours, &params, &obvs, mosaic);
		} else if(MOD==1){												// Evolution in a quenched potential
			evolve_quenched(&phi, &y_colloid, &params, &obvs, mosaic);
		} else {
			printhelp(&params);
            exit(EXIT_FAILURE);
		}
		
		flag = params.mc_runs/10;										// Prints completing percentage
		if( flag !=0 && ((mc_counter +1)%flag) == 0 ) printf("# MC PROGRESS %d%%\n", (mc_counter +1)/flag*10);
	}
	
	//print_observables(&obvs, &params);
	print_trajectory(&obvs, &params);
	
	// Free memory and exit
	free(phi);
	free(y_colloid);
	for(i=0; i<n_sites; i++) free(neighbours[i]);
	for(i=0; i<n_sites; i++) free(mosaic[i]);
	free(obvs.colloid_pos);												// Add any other observable you are storing
	return 0;
}


// DEFINITION OF FUNCTIONS

// Default parameters
void default_parameters(parameter* params){
	params->mass = 0.0;
	params->lambda = 0.25;
	params->quartic_u = 0.0;
	params->temperature = 0.001;
	params->relativeD = 1.0;
	params->trap_strength = 0.1;
	params->rng_seed = -1; 												// if seed is -1 (not given by user), it will be picked randomly
	params->system_size = 128;
	params->delta_t = 0.01;
	params->n_timestep = 100000;
	params->mc_runs = 10000;
	params->R = 1;
	params->X0 = 2;
}

// All that needs to be done once
void initialise(long*** neighbours, parameter* params, long*** mosaic){
	// i) GSL random number generator setup
	gsl_rng_env_setup();
    T = gsl_rng_default;
    seed = gsl_rng_alloc (T);											// This is probably not really a seed, but sticazzi
    time_t t;															// If no seed provided, draw a random one
    if(params->rng_seed==-1) params->rng_seed = (unsigned) time(&t) % 100000; 
    gsl_rng_set(seed, params->rng_seed);
	
	// ii) What are each position's neighbours?
	long L = params->system_size;
	neighborhood(neighbours, L);
	
	// iii) Initialize the mosaic
	double dist;
	long n_sites = intpow(L, DIM);
	int i, j, d;
	for(i=0; i<n_sites; i++){
		top=0;															// Number of sites within 5R around a given site (global variable)
		for(j=0; j<n_sites; j++){
			dist=0;
			for(d=0; d<DIM; d++) dist += floatpow( modular( ind2coord(d,i,L), ind2coord(d,j,L), L) , 2);
			dist = sqrt(dist);
			if(dist < ceil(5*params->R) + EPS ){
				(*mosaic)[i][top]=j;
				top++;
			}
		}
	}								
}

// Reset field and colloid to initial conditions
void wipe(double** phi, double** y_colloid, parameter* params)
{
	int n_sites, i;
	n_sites = intpow(params->system_size, DIM);
	for(i = 0; i < n_sites; i++){ 
		 //(*phi)[i] = gsl_ran_gaussian_ziggurat(seed,1.0);				// Infinite temperature state
		 (*phi)[i] = 0;													// Initially flat field - default after calloc()
		 //(*phi)[i] = (i % 2 ? 1 : -1);								// Staggered
	}
	
	// y - colloid (initially in the middle of the lattice, where the harmonic well stands)
	long L = params->system_size;
	for(i=0; i<DIM; i++) (*y_colloid)[i] = L/2;
}

// This initialises the observables structure later containing the measurements
void initialise_observable(observables* obvs, parameter* params)
{
	obvs->write_time_delta = 0.2; 										// This is in physical time units, so writing occurs every (write_time_delta / n_timestep) integration step
	
	// This counts how many writing events will occur in time (including t=0, thus + 1). If you choose exp distributed measurements, it still works (but it's overestimated).
	int writing_times = (int)(1 + (((params->n_timestep)*params->delta_t)/obvs->write_time_delta)); 

	//CALLOC(obvs->colloid_msd, writing_times); 						// Save MSD vs time
	CALLOC(obvs->colloid_pos, writing_times); 							// Save position vs time
	
	/*
	// FIELD MEASUREMENTS ONLY
	CALLOC(obvs->field_average, writing_times); 						// Prepare writing_times many arrays to store averages
	//CALLOC(obvs->field_correlation, writing_times); 					// Prepare writing_times many arrays to store correlations
	int i;
	for(i = 0; i < writing_times; i++)
	{
		//obvs->field_correlation[i] = calloc( params->system_size , sizeof(double) );
		//if( (obvs->field_correlation[i]) == NULL){printf("Allocation of '(obvs->field_correlation[%i])'  failed. Terminate. \n", i); exit(2);} 
		obvs->field_average[i] = calloc( params->system_size , sizeof(double) );
		if( (obvs->field_average[i]) == NULL){
			printf("Allocation of '(obvs->field_average[%i])'  failed. Terminate. \n", i); 
			exit(2);
		} 
	}
	*/
}

// Perform measurements and print them out
void measure(double** phi, double** y_colloid, long tstep, parameter* params, observables* obvs){
	long i;
	printf("%g\t", tstep*params->delta_t);
	for(i = 0; i < params->system_size; i++){
		obvs->field_average[obvs->write_count][i] += (*phi)[i];
		obvs->field_correlation[obvs->write_count][i] += ((*phi)[0] * (*phi)[i]);
	}	
	
	long n_sites = intpow(params->system_size , DIM);
	for(i = 0; i < DIM; i++){
		obvs->colloid_msd[obvs->write_count] += ((*y_colloid)[i] - ind2coord(i, n_sites/2 , params->system_size))*((*y_colloid)[i] - ind2coord(i, n_sites/2 , params->system_size));
	}
	//printf("\n");
}

// Creates list of nearest neighbours in DIM dimensions
void neighborhood(long*** list, int L){
	int k,d;
	int vec[DIM], neigh[DIM];

	for(k=0; k<intpow(L,DIM); k++){										// Cycles over lattice sites
		for(d=0; d<DIM; d++){											// Finds the DIM coordinates of the current site
			vec[d] = ind2coord(d,k,L);
			neigh[d] = vec[d];
		}
		for(d=0; d<DIM; d++){
			neigh[d] = (vec[d]+1)%L;									// Finds right neighbour
			(*list)[k][2*d] = vec2ind(neigh,L);							// Stores it in the neighbour list
			neigh[d] = (vec[d]+L-1)%L;									// Finds left neighbour
			(*list)[k][2*d+1] = vec2ind(neigh,L);						// Stores it in the neighbour list
			neigh[d] = vec[d];											// Restores local copy (prepares for next dimension)
		}
	}
}

// Pre-thermalization of the field
void prethermalize(double** phi, parameter* params){
	// Benjamin: 20/08/21, spectral thermalisation
	// \phi_k are complex normal random variables with <\phi_k> = 0, < |\phi_k|^2> = k_B T*(k^2+r)^-1, and \phi_k = \phi^*_{-k}
	// Using FFTW library, -lfftw3
	fftw_complex *phi_fourier;
	fftw_plan p;

	int n_sites = intpow(params->system_size, DIM);
	int i, j;
	
	phi_fourier = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (n_sites/2 + 1)); // complex Hermitian array
	int *vec = malloc(DIM * sizeof(int));
	/* fill with random data */
	double inverse_variance, variance;
	double inv_L_square = (39.438/pow(((double) params->system_size), 2)); //(2\pi /L)^2
	double radius;
	double polar_angle;

	int centre = (n_sites/2);
	for(i = centre; i >=0 ; i--) // Fill half the reciprocal lattice without the zero mode first. The first entry is at the centre of the reciprocal lattice and therefore the zero mode (~ k_BT/r)
	{
		// What's the L2 norm of the reciprocal vector?
		inverse_variance = params->mass;
		ind2vecROW(&vec, i, DIM, params->system_size);
		for(j = 0; j < DIM; j++)
		{
			inverse_variance += inv_L_square*(vec[j]*vec[j]);
		}
		variance = (params->temperature/inverse_variance); // k_B T /(q^2 + r)

		if(inverse_variance > 0){
			radius = gsl_ran_gaussian_ziggurat(seed, sqrt(variance)); 
			polar_angle =  gsl_ran_flat(seed, 0, 6.2831); // pick random angle
		
			phi_fourier[centre-i][0] = radius*cos(polar_angle);
			phi_fourier[centre-i][1] = radius*sin(polar_angle);
		}
		else{ // Critical mode is force set to zero
			phi_fourier[centre-i][0] = 0.0;
			phi_fourier[centre-i][1] = 0.0;
		}
	}
	
	// _c2r transforms a half-complex array into real fft http://www.fftw.org/fftw3_doc/Real_002ddata-DFTs.html
	int lattice_dimensions[DIM];
	for(i = 0; i < DIM; i++){lattice_dimensions[i] = ((int) params->system_size);} //It's a cube
	
	double* phi_real;
 	phi_real = (double*)fftw_malloc(n_sites * sizeof(double));

	// http://www.fftw.org/fftw3_doc/Real_002ddata-DFTs.html
	//fftw_plan fftw_plan_dft_c2r(int rank, const int *n, fftw_complex *in, double *out, unsigned flags);
	p = fftw_plan_dft_c2r(DIM, lattice_dimensions, phi_fourier, phi_real,  FFTW_ESTIMATE);
	if(p == NULL){printf("fftw plan didn't work\n");}
	fftw_execute(p); 
	
	// normalising inverse transform with V^(-d/2), ass fftw3 doesn't do that.
	double normalising_volume = (1.0/(((double) n_sites))); 

	// Now transform rowmajor phi_real into col_major *phi -> matrix transpose
	/* Only for D > 1
	 * int row_maj_index, col_maj_index, dir;
	for(row_maj_index = 0; row_maj_index < n_sites; row_maj_index++)
	{
		for(dir = 0; dir < DIM; dir++)
		{
			vec[dir] = ((int) (row_maj_index / intpow(params->system_size, (DIM - dir -1))) % params->system_size);
		}
	       	col_maj_index = vec2ind(vec, params->system_size);
		if(row_maj_ind
		printf("RM %i CM %i | phi_real: %g\n", row_maj_index, col_maj_index, phi_real[row_maj_index]);
		(*phi)[col_maj_index] = phi_real[row_maj_index]; 
	}*/

	// For D = 1
	// Shift origin to centre
	for(i = 0; i <= centre; i++) (*phi)[i] = normalising_volume*phi_real[centre-i];
	for(i = centre+1; i < n_sites; i++) (*phi)[i] = normalising_volume*phi_real[n_sites+centre-i];
	
    fftw_free(phi_fourier); 
    fftw_free(phi_real); 
	fftw_destroy_plan(p);
}


// Preparation of the field in equilibrium at T=0 around X0 
// 1d only at the moment. It can probably be generalized to finite T by adding noise later on.
void field_prepare(double** phi, parameter* params){
	// Spectral initialization using FFTW library, -lfftw3
	// \phi_k are complex normal random variables with <\phi_k> = 0, < |\phi_k|^2> = k_B T*(k^2+r)^-1, and \phi_k = \phi^*_{-k}
	fftw_complex *phi_fourier;
	fftw_plan p;

	int n_sites = intpow(params->system_size, DIM);
	int i, j;
	
	phi_fourier = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (n_sites/2 + 1)); // complex Hermitian array
	int *vec = malloc(DIM * sizeof(int));
	/* fill with random data */
	//double inverse_variance, variance;
	double inverse_variance;
	double inv_L = 6.28318530718/params->system_size;									// 2pi/L			
	double inv_L_square = (39.438/intpow(params->system_size, 2)); 						// (2\pi /L)^2
	double radius;
	double polar_angle;

	int centre = (n_sites/2);
	for(i = centre; i >=0 ; i--) // Fill half the reciprocal lattice without the zero mode first. The first entry is at the centre of the reciprocal lattice and therefore the zero mode (~ k_BT/r)
	{
		// What's the L2 norm of the reciprocal vector?
		inverse_variance = params->mass;
		ind2vecROW(&vec, i, DIM, params->system_size);
		for(j = 0; j < DIM; j++)
		{
			inverse_variance += inv_L_square*(vec[j]*vec[j]);
		}
		//variance = (params->temperature/inverse_variance); // k_B T /(q^2 + r)
		radius = params->lambda * exp(- floatpow(params->R,2)*(inverse_variance - params->mass)/2) /inverse_variance ; // l* exp(-q^2 R^2/2) / (q^2 + r)

		if(inverse_variance > 0){
			//radius = gsl_ran_gaussian_ziggurat(seed, sqrt(variance)); 
			//polar_angle =  gsl_ran_flat(seed, 0, 6.2831); 							// pick random angle
			polar_angle = - inv_L * vec[j] * (params->system_size/2+params->X0);		// Careful if you ever generalize to higher dimensions
		
			phi_fourier[centre-i][0] = radius*cos(polar_angle);
			phi_fourier[centre-i][1] = radius*sin(polar_angle);
		}
		else{ // Critical mode is force set to zero
			phi_fourier[centre-i][0] = 0.0;
			phi_fourier[centre-i][1] = 0.0;
		}
	}

	// _c2r transforms a half-complex array into real fft http://www.fftw.org/fftw3_doc/Real_002ddata-DFTs.html
	int lattice_dimensions[DIM];
	for(i = 0; i < DIM; i++){lattice_dimensions[i] = ((int) params->system_size);} //It's a cube
	
	double* phi_real;
 	phi_real = (double*)fftw_malloc(n_sites * sizeof(double));

	// http://www.fftw.org/fftw3_doc/Real_002ddata-DFTs.html
	//fftw_plan fftw_plan_dft_c2r(int rank, const int *n, fftw_complex *in, double *out, unsigned flags);
	p = fftw_plan_dft_c2r(DIM, lattice_dimensions, phi_fourier, phi_real,  FFTW_ESTIMATE);
	if(p == NULL){printf("fftw plan didn't work\n");}
	fftw_execute(p); 
	
	// normalising inverse transform with V^(-d/2), ass fftw3 doesn't do that.
	double normalising_volume = (1.0/(((double) n_sites))); 

	// Now transform rowmajor phi_real into col_major *phi -> matrix transpose
	/* Only for D > 1
	 * int row_maj_index, col_maj_index, dir;
	for(row_maj_index = 0; row_maj_index < n_sites; row_maj_index++)
	{
		for(dir = 0; dir < DIM; dir++)
		{
			vec[dir] = ((int) (row_maj_index / intpow(params->system_size, (DIM - dir -1))) % params->system_size);
		}
	       	col_maj_index = vec2ind(vec, params->system_size);
		if(row_maj_ind
		printf("RM %i CM %i | phi_real: %g\n", row_maj_index, col_maj_index, phi_real[row_maj_index]);
		(*phi)[col_maj_index] = phi_real[row_maj_index]; 
	}*/

	// For D = 1
	// Shift origin to centre
	for(i = 0; i <= centre; i++) (*phi)[i] = normalising_volume*phi_real[centre-i];
	for(i = centre+1; i < n_sites; i++) (*phi)[i] = normalising_volume*phi_real[n_sites+centre-i];

    fftw_free(phi_fourier); 
    fftw_free(phi_real); 
	fftw_destroy_plan(p);
}

// Evolution - colloid only (quenched potential)
void evolve_quenched(double** phi, double** y_colloid, parameter* params, observables* obvs, long** mosaic){
	// Preparing variables for field evolution (Euler-Maruyama)
	long tstep, n_timestep;
	n_timestep = params->n_timestep;
	
	// Preparing variables for colloid evolution (Stochastic Runge-Kutta II)
	int i, j, y_site;
	double grad, noise, F2, r, weight;	
	double w[DIM], F1[DIM], Y[DIM];
	long L = params->system_size;
	double noise_intensity = sqrt(2.0 * params->temperature * params->relativeD * params->delta_t);
	for(j=0; j<DIM; j++) w[j] = L/2;												// Finds position of the harmonic well
	
	// Preparing measurement process
	int write_time_delta_step = (int) ((obvs->write_time_delta)/(params->delta_t));
	int next_writing_step = 0;
	obvs->write_count = 0;
	
	for(tstep = 0; tstep < n_timestep; tstep++){									// Time evolution
	
		// MEASUREMENT
		if(tstep == next_writing_step){
			obvs->colloid_pos[obvs->write_count] += (*y_colloid)[0];				// Save colloid position
			//obvs->colloid_msd[obvs->write_count] += (*y_colloid)[1];				// Save colloid position of 2nd coordinate
			next_writing_step += write_time_delta_step;								// Linearly distributed
			obvs->write_count++;
		}
		
		// EVOLUTION

		// i) Create local copy of the colloid variable
		for(i=0; i<DIM; i++) Y[i] = (*y_colloid)[i];	
		
		// ii) Prediction step for the colloid
		y_site = closest_site(Y,L);													// Finds the lattice site closest to Y
		
		for(i=0; i<DIM; i++){ 														// Evolves each of the components separately
			// Compute gradient of the field under the colloid
			grad = 0;
			for(j=0; j<top; j++){
				r = distance(Y, mosaic[y_site][j], L);
				weight = gaussian(r,params->R);
				grad += 0.5 * ( (*phi)[mosaic[y_site][j]+intpow(L,i)] - (*phi)[mosaic[y_site][j]-intpow(L,i)] ) * weight;
			}
			
			// Compute temporary position
			noise = noise_intensity * gsl_ran_gaussian_ziggurat(seed, 1.0);
			F1[i] =  params->relativeD * (params->lambda*grad -params->trap_strength*( Y[i]-w[i] ));
			(*y_colloid)[i] += params->delta_t * F1[i] + noise;
		}

		// iv) Correction step for the colloid (with the evolved field)
		y_site = closest_site(*y_colloid,L);										// Compute new site of the colloid
		
		for(i=0; i< DIM; i++){														// Evolve each of the components separately
			// Compute new gradient
			grad = 0;
			for(j=0; j<top; j++){
				r = distance(*y_colloid, mosaic[y_site][j], L);
				weight = gaussian(r,params->R);
				grad += 0.5 * ( (*phi)[mosaic[y_site][j]+intpow(L,i)] - (*phi)[mosaic[y_site][j]-intpow(L,i)] ) * weight;
			}
			
			// Compute corrected contribution
			F2 = params->relativeD * ( params->lambda*grad -params->trap_strength*( (*y_colloid)[i]-w[i] ) );
			
			// Sum the two contributions: y_n+1 = y_n + 1/2(F1+F2)*dt + noise
			noise = noise_intensity * gsl_ran_gaussian_ziggurat(seed, 1.0);
			(*y_colloid)[i] = Y[i] + 0.5*(F1[i]+F2)*params->delta_t + noise;
		}
	}
}

// Model B evolution
void evolveB(double** phi, double** y_colloid, long** neighbours, parameter* params, observables* obvs, long** mosaic){
	// Preparing variables for field evolution (Euler-Maruyama)
	long tstep, n_sites, n_timestep;
	n_sites = intpow(params->system_size, DIM);
	n_timestep = params->n_timestep;
	
	// TODO it's a terrible idea to allocate memory here, because this function is called M times
	double* laplacian_phi;
	ALLOC(laplacian_phi, n_sites);
	double* laplacian_square_phi;
	ALLOC(laplacian_square_phi, n_sites);
	double* laplacian_phi_cubed;
	CALLOC(laplacian_phi_cubed, n_sites);
	double* noise_field;
	CALLOC(noise_field, DIM * n_sites);
	double* noise_gradient;
	CALLOC(noise_gradient, n_sites);
	
	// Preparing variables for colloid evolution (Stochastic Runge-Kutta II)
	int i, j, y_site;
	double grad, noise, F2, r, weight;	
	double w[DIM], F1[DIM], Y[DIM];
	long L = params->system_size;
	double noise_intensity = sqrt(2.0 * params->temperature * params->relativeD * params->delta_t);
	for(j=0; j<DIM; j++) w[j] = L/2;												// Finds position of the harmonic well
	
	// Preparing measurement process
	int write_time_delta_step = (int) ((obvs->write_time_delta)/(params->delta_t));
	int next_writing_step = 0;
	obvs->write_count = 0;
	
	for(tstep = 0; tstep < n_timestep; tstep++){									// Time evolution
	
		// MEASUREMENT
		
		if(tstep == next_writing_step){
			//measure(phi, y_colloid, tstep, params, obvs); 						// Evaluate all sorts of correlators etc. here
			//for(i = 0; i < params->system_size; i++) obvs->field_average[obvs->write_count][i] += (*phi)[i]; // Save field average
			obvs->colloid_pos[obvs->write_count] += (*y_colloid)[0];				// Save colloid position
			//obvs->colloid_msd[obvs->write_count] += (*y_colloid)[1];				// Save colloid position of 2nd coordinate
			next_writing_step += write_time_delta_step;								// Linearly distributed
			obvs->write_count++;
		}
		
		// EVOLUTION

		// i) Create local copy of the colloid variable
		for(i=0; i<DIM; i++) Y[i] = (*y_colloid)[i];	
		
		// ii) Prediction step for the colloid
		y_site = closest_site(Y,L);													// Finds the lattice site closest to Y
		
		for(i=0; i<DIM; i++){ 														// Evolves each of the components separately
			// Compute gradient of the field under the colloid
			grad = 0;
			for(j=0; j<top; j++){
				r = distance(Y, mosaic[y_site][j], L);
				weight = gaussian(r,params->R);
				grad += 0.5 * ( (*phi)[mosaic[y_site][j]+intpow(L,i)] - (*phi)[mosaic[y_site][j]-intpow(L,i)] ) * weight;
			}
			
			// Compute temporary position
			noise = noise_intensity * gsl_ran_gaussian_ziggurat(seed, 1.0);
			F1[i] =  params->relativeD * (params->lambda*grad -params->trap_strength*( Y[i]-w[i] ));
			(*y_colloid)[i] += params->delta_t * F1[i] + noise;
		}
		
		// iii) Evolve the field with the local copy of the colloid position
		laplacian(&laplacian_phi, *phi, neighbours, n_sites);						// Write Laplacian of phi into laplacian_phi
		laplacian(&laplacian_square_phi, laplacian_phi, neighbours, n_sites);		// Write (D^2)^2 phi into laplacian_square_phi
		if(params->quartic_u > EPS){
			laplacian_of_cube(&laplacian_phi_cubed, *phi, neighbours, n_sites);		// Write D2 [phi(x)^3] into laplacian_phi_cubed
		}
		if(params->temperature > EPS){
			generate_noise_field(&noise_field, DIM*n_sites, params);				// Fill \vec{Lambda} with randomness
			gradient_field(&noise_gradient, noise_field, neighbours, n_sites);		// Compute gradient noise term
		}	

		// Add together to new step d/dt phi = -a * D2 phi - b D4 phi - u D2 (phi^3) + D * noise (D is nabla)
		phi_evolveB(phi, laplacian_phi, laplacian_square_phi, laplacian_phi_cubed, noise_gradient, n_sites, params, Y, mosaic);

		// iv) Correction step for the colloid (with the evolved field)
		y_site = closest_site(*y_colloid,L);										// Compute new site of the colloid
		
		for(i=0; i< DIM; i++){														// Evolve each of the components separately
			// Compute new gradient
			grad = 0;
			for(j=0; j<top; j++){
				r = distance(*y_colloid, mosaic[y_site][j], L);
				weight = gaussian(r,params->R);
				grad += 0.5 * ( (*phi)[mosaic[y_site][j]+intpow(L,i)] - (*phi)[mosaic[y_site][j]-intpow(L,i)] ) * weight;
			}
			
			// Compute corrected contribution
			F2 = params->relativeD * ( params->lambda*grad -params->trap_strength*( (*y_colloid)[i]-w[i] ) );
			
			// Sum the two contributions: y_n+1 = y_n + 1/2(F1+F2)*dt + noise
			noise = noise_intensity * gsl_ran_gaussian_ziggurat(seed, 1.0);
			(*y_colloid)[i] = Y[i] + 0.5*(F1[i]+F2)*params->delta_t + noise;
		}
	}
	
	// Free memory
	free(laplacian_phi);
	free(laplacian_square_phi);
	free(laplacian_phi_cubed);
	free(noise_field);
	free(noise_gradient);
}

// Model A evolution
void evolveA(double** phi, double** y_colloid, long** neighbours, parameter* params, observables* obvs, long** mosaic){
	// Preparing variables for field evolution (Euler-Maruyama)
	long tstep, n_sites, n_timestep;
	n_sites = intpow(params->system_size, DIM);
	n_timestep = params->n_timestep;
	
	double* laplacian_phi;
	ALLOC(laplacian_phi, n_sites);
	double* noise_field;
	CALLOC(noise_field, n_sites);

	int i, j, y_site;
	double grad, noise, F2, r, weight;
	double w[DIM], F1[DIM], Y[DIM];
	long L = params->system_size;
	double noise_intensity = sqrt(2.0 * params->temperature * params->relativeD * params->delta_t);
	
	for(i=0; i<DIM; i++) w[i] = L/2;												// Finds position of the harmonic well
	
	// Preparing measurement process
	int write_time_delta_step = (int) ((obvs->write_time_delta)/(params->delta_t));
	int next_writing_step = 0;														// Linearly distributed writing times
	//int next_writing_step = write_time_delta_step;								// Exponentially distributed writing times TODO
	obvs->write_count = 0;
	
	for(tstep = 0; tstep < n_timestep; tstep++){									// Time evolution
	
		// MEASUREMENT
		
		if(tstep == next_writing_step){
			//measure(phi, y_colloid, tstep, params, obvs); 						// Evaluate all sorts of correlators etc. here
			obvs->colloid_pos[obvs->write_count] += (*y_colloid)[0];				// Save colloid position
			//for(i = 0; i < params->system_size; i++) obvs->field_average[obvs->write_count][i] += (*phi)[i]; // Save field average
			//obvs->colloid_msd[obvs->write_count] += (*y_colloid)[1];				// Save colloid position of 2nd coordinate
			next_writing_step += write_time_delta_step;								// Linearly distributed
			//next_writing_step = (int)(next_writing_step * exp( write_time_delta_step));	// Exponentially distributed TODO
			obvs->write_count++;
		}
		
		// EVOLUTION
									
		// i) Create local copy of the colloid variable
		for(i=0; i<DIM; i++) Y[i] = (*y_colloid)[i];	
		
		// ii) Prediction step for the colloid
		y_site = closest_site(Y,L);													// Finds the lattice site closest to Y
		
		for(i=0; i<DIM; i++){ 														// Evolve each of the components separately
			// Compute gradient of the field under the colloid
			grad = 0;
			for(j=0; j<top; j++){
				r = distance(Y, mosaic[y_site][j], L);
				weight = gaussian(r,params->R);
				grad += 0.5 * ( (*phi)[mosaic[y_site][j]+intpow(L,i)] - (*phi)[mosaic[y_site][j]-intpow(L,i)] ) * weight;
			}
			
			// Compute temporary position
			noise = noise_intensity * gsl_ran_gaussian_ziggurat(seed, 1.0);
			F1[i] =  params->relativeD * (params->lambda*grad -params->trap_strength*( Y[i]-w[i] ));
			(*y_colloid)[i] += params->delta_t * F1[i] + noise;
		}
	
		// iii) Evolve the field with the local copy of the colloid position	
		laplacian(&laplacian_phi, *phi, neighbours, n_sites);						// Write Laplacian of phi into laplacian_phi
		if(params->temperature > EPS){
			generate_noise_field(&noise_field, n_sites, params);					// Fill Lambda with randomness
		}
				
		// Add together to new step d/dt phi = -r * phi + \nabla^2 phi - u * (phi^3) + l * V(x-Y) + noise
		phi_evolveA(phi, laplacian_phi, noise_field, n_sites, params, Y, mosaic);
		
		// iv) Correction step for the colloid (with the evolved field)
		y_site = closest_site(*y_colloid,L);										// Compute new site of the colloid
		 	
		for(i=0; i< DIM; i++){														// Evolve each of the components separately
			// Compute new gradient
			grad = 0;
			for(j=0; j<top; j++){
				r = distance(*y_colloid, mosaic[y_site][j], L);
				weight = gaussian(r,params->R);
				grad += 0.5 * ( (*phi)[mosaic[y_site][j]+intpow(L,i)] - (*phi)[mosaic[y_site][j]-intpow(L,i)] ) * weight;
			}
						
			// Compute corrected contribution
			F2 = params->relativeD * (params->lambda*grad -params->trap_strength*( (*y_colloid)[i]-w[i] ));
			
			// Sum the two contributions: y_n+1 = y_n + 1/2(F1+F2)*dt + noise
			noise = noise_intensity * gsl_ran_gaussian_ziggurat(seed, 1.0);
			(*y_colloid)[i] = Y[i] + 0.5*(F1[i]+F2)*params->delta_t + noise;
		}
	}
	
	// Free memory
	free(laplacian_phi);
	free(noise_field);
}

// Field evolution - Model B
void phi_evolveB(double** phi, double* laplacian_phi, double* laplacian_square_phi, double* laplacian_phi_cubed, double* noise_gradient, long n_sites, parameter* params, double* Y, long** mosaic){
	long i;
	long L = params->system_size;
	double delta_t = params->delta_t;
	double r, weight;

	for(i = 0; i < n_sites; i++){
		(*phi)[i] += (delta_t*( -laplacian_square_phi[i] + params->mass*laplacian_phi[i] + params->quartic_u * laplacian_phi_cubed[i]) + noise_gradient[i]);
	}
	
	// Interaction with the colloid
	int y_site = closest_site(Y,L);													// Finds the lattice site closest to Y, the colloid
	for(i=0; i<top; i++){
		r = distance(Y, mosaic[y_site][i], L);
		weight = laplacian_gaussian(r,params->R);
		(*phi)[mosaic[y_site][i]] -= delta_t * params->lambda * weight;
	}
}

// Field evolution - Model A
void phi_evolveA(double** phi, double* laplacian_phi, double* noise_field, long n_sites, parameter* params, double* Y, long** mosaic){
	long i;
	double delta_t = params->delta_t;
	long L = params->system_size;
	double r, weight;
	
	for(i = 0; i < n_sites; i++){													// Notice noise_field contains delta_t in its variance
		(*phi)[i] += delta_t*(- params->mass*(*phi)[i] + laplacian_phi[i] - params->quartic_u * floatpow((*phi)[i],3) ) + noise_field[i];
	}
	
	// Interaction with the colloid
	int y_site = closest_site(Y,L);													// Finds the lattice site closest to Y, the colloid
	for(i=0; i<top; i++){
		r = distance(Y, mosaic[y_site][i], L);
		weight = gaussian(r,params->R);
		(*phi)[mosaic[y_site][i]] += delta_t * params->lambda * weight;
	}
}

// Returns Laplacian as calculated from cubic neighbour cells in DIM dimensions
void laplacian(double** laplacian,  double* field, long** neighbours, long n_sites){
	double buffer;
	long pos, i;
	for(pos = 0; pos < n_sites; pos++){
		buffer = 0;
		for(i = 0; i < 2*DIM; i++) {buffer += field[neighbours[pos][i]]; }
		buffer -= (2*DIM*field[pos]);
		(*laplacian)[pos] = buffer;
	}
}

// Returns the Laplacian of field^3
void laplacian_of_cube(double** laplacian,  double* field, long** neighbours, long n_sites){
	double buffer;
	long pos, i;
	for(pos = 0; pos < n_sites; pos++){
		buffer = 0;
		for(i = 0; i < 2*DIM; i++){
			buffer += pow( (field[neighbours[pos][i]]), 3); 
		}
		buffer -= (2*DIM*field[pos]*field[pos]*field[pos]);
		(*laplacian)[pos] = buffer;
	}
}

// This function generates a completely uncorrelated random field on a line
void generate_noise_field(double** noise_field, long length, parameter* params){
	long i;
	double noise_intensity = sqrt(2.0 * params->temperature * params->delta_t);
	for(i = 0; i < length; i++){
		(*noise_field)[i] = noise_intensity * gsl_ran_gaussian_ziggurat(seed, 1.0);
	}
}

// Computes the gradient of a (noisy) field
void gradient_field(double** grad_noise, double* noise, long** neighbours, long n_sites){
	long i;
	int j; // neighbour of i
	double buffer;
	for(i = 0; i < n_sites; i++){
		buffer = 0;
		for(j = 0; j < DIM; j++){
			buffer += noise[neighbours[i][2*j]];
			buffer -= noise[neighbours[i][2*j+1]];
		}
		(*grad_noise)[i] = 0.5*buffer;
	}
}

// Returns the value of a Gaussian of variance R evaluated at r
double gaussian(double r, double R){
	return gauss_norm * exp(-r*r/(2*R*R));
}

// Returns the value of the laplacian of a Gaussian of variance R evaluated at r
double laplacian_gaussian(double r, double R){
	return gauss_norm * exp(-r*r/(2*R*R)) * (r*r - DIM*R*R) / floatpow(R,4);
}

// Translate from a list index (k) to d-th lattice index, and viceversa.
inline int ind2coord(int d, int k, int L) {return ( (int)( k/intpow(L,d) ) )%L ;}
int vec2ind(int *vec, int L){
	int i, res=0;
	for(i=0; i<DIM; i++) res += vec[i] * intpow(L,i);
	return res;
}

void ind2vecROW(int** vec, int ind, int dim, int L)
{	
	// Places origin in the middle, row-major format.
	int dir;
	int offset = (int) ((L-1)/2.0); // So if L = 7 then the middle is at ___3___ 
	for(dir = 0; dir < dim; dir++)
	{
		(*vec)[dir] = (((int) (ind / intpow(L, (dim - dir -1))) % L) - offset) ; // row format, last index is fastest moving
	}
}

// Finds index of closest lattice site to the vector "vec" 
// I'm using a special modulo function to avoid getting negative return values, eg -3 % 10 = -3, but modulo(-3,10) = 7.
int closest_site(double *vec, int L){
	int i, site=0;
	for(i=0; i<DIM; i++){	
		site += (int)(modulo( round(vec[i])  , L) * intpow(L,i)); 
	}
	return site;
}

// According to https://stackoverflow.com/questions/14997165/fastest-way-to-get-a-positive-modulo-in-c-c this is still fast
unsigned modulo( int value, unsigned m) {
    int mod = value % (int)m;
    if (mod < 0) {
        mod += m;
    }
    return mod;
}

// Return the distance mod(L)
double modular(double a, double b, double L){
    return fabs(L/2 - fmod(3*L/2 + a - b, L));
}

// Returns the distance between the position vector and a given site (in a lattice of side L)
double distance(double* pos, int site, int L){
	int d;
	double dist=0;
	//for(d=0; d<DIM; d++) dist += floatpow( pos[d] - ind2coord(d,site,L) ,2);
	for(d=0; d<DIM; d++) dist += floatpow( modular(pos[d], ind2coord(d,site,L), L) , 2);
	dist = sqrt(dist);
	return dist;
}

// Factorial of a number
int factorial(int n){
    if (n == 0) return 1;
    return n * factorial(n - 1);
}

// Double factorial of a number
int double_factorial(int n){
    if (n <= 1) return 1;
    return n * factorial(n - 2);
}

// The usual pow(a,b)=exp(log(a) * b) is slow AF
int intpow(int a, int b){
	int i, res=1;
	for(i=0; i<b; i++) res *= a;
	return res;
} 

// The usual pow(a,b)=exp(log(a) * b) is slow AF
double floatpow(double a, int b){
	int i;
	double res=1;
	for(i=0; i<b; i++) res *= a;
	return res;
}

// Prints observables including field
void print_observables(observables* obvs, parameter* params){
	int i,j;
	int system_size = params->system_size;
	int write_count = obvs->write_count;
	double weight = 1/((double) params->mc_runs);

	for(i = 0; i < write_count; i ++){
		printf("#FIELDAVG %g", i * (obvs->write_time_delta));
		for(j = 0; j < system_size; j++){
			printf("\t%g", weight*(obvs->field_average[i][j]));
		}
		printf("\n");
	}

	/*
	for(i = 0; i < write_count; i ++){
		printf("#FIELDCORR %g", i * (obvs->write_time_delta));
		for(j = 0; j < system_size; j++){
			printf("\t%g", weight*(obvs->field_correlation[i][j]- (obvs->field_average[i][0] * obvs->field_average[i][j]) ));
		}
		printf("\n");
	}

	// Output MSD of colloid
	for(i = 0; i < write_count; i++){
		printf("# COLLOIDMSD %g", i * (obvs->write_time_delta));
		printf("\t%.3f",weight*(obvs->colloid_msd[i]));
		printf("\n");
	}
	*/
	
	// Output trajectory of colloid
	for(i = 0; i < write_count; i++){
		printf("#COLLOID_X %g", i * (obvs->write_time_delta));
		//printf("%.4f", i * (obvs->write_time_delta));
		printf("\t%.15f",weight*obvs->colloid_pos[i]- params->system_size/2);
		//printf("\t%.12f",weight*(obvs->colloid_msd[i]));
		printf("\n");
	}
}

// Prints colloid-related observables only
void print_trajectory(observables* obvs, parameter* params)
{
	int i;
	int write_count = obvs->write_count;
	double weight = 1/((double) params->mc_runs);

	// Output trajectory of colloid
	for(i = 0; i < write_count; i++){
		//printf("#COLLOID_X %g", i * (obvs->write_time_delta));
		printf("%.4f", i * (obvs->write_time_delta));
		printf("\t%.15f", weight*obvs->colloid_pos[i]- params->system_size/2);
		//printf("\t%.12f",weight*(obvs->colloid_msd[i]));
		printf("\n");
	}
}

// Prints out simulation parameters
void print_params(parameter* params){
printf("# Parameters\n\
# MASS %g\n\
# LAMBDA (field-colloid-coupling) %g\n\
# U (quartic coupling) %g\n\
# TEMPERATURE %g\n\
# RELATIVE MOTILITY D %g\n\
# TRAP STRENGTH %g\n\
# RNG SEED %u\n\
# L (System Size) %i\n\
# DIM %i\n\
# RADIUS %g\n\
# INITIAL DISPLACEMENT %g\n\
# DELTA T %g\n\
# TIMESTEPS %lu\n\
# MONTE CARLO RUNS %i\n",
params->mass, params->lambda, params->quartic_u, params->temperature, params->relativeD, params->trap_strength, params->rng_seed, params->system_size, DIM, params->R, params->X0, params->delta_t, params->n_timestep, params->mc_runs);
		
	if(MOD==0) printf("# MODEL A\n");
	if(MOD==2) printf("# MODEL B\n");
}

// Prints out the instructions
void printhelp(parameter* params){
	printf("# Colloid in Gaussian Field\n# '%s' built %s\n\
# Use with options flags\n\
# -L Length of d-dimensional lattice (default %i)\n\
# -r Mass of Gaussian field (default %g)\n\
# -l Coupling strength between colloid and field (default %g)\n\
# -u Quartic coupling strength (default %g)\n\
# -T Temperature of bath (default %g)\n\
# -n Relative motility colloid/field (default %g)\n\
# -k Strength of harmonic trap (default %g)\n\
# -S Seed for RNG (default random)\n\
# -t Integration timestep (default %g)\n\
# -d Dimension (default %i)\n\
# -N Number of timesteps (default %lu)\n\
# -M Number of Monte Carlo samples (default %i)\n\
# -R Radius of the colloid (default %g)\n\
# -X Initial displacement of the colloid (default %g)\n\
# -m Field dynamics (model A -> 0, model B -> 2, quenched -> 1, default %i)\n\
# -h To see this helpscreen\n\
# -P Output source code\n",__FILE__,__DATE__, params->system_size, params->mass, params->lambda, params->quartic_u, params->temperature, params->relativeD, params->trap_strength, params->delta_t, DIM, params->n_timestep, params->mc_runs, params->R, params->X0, MOD);
}

// Prints out the nearest-neighbour list
// For neighbour list, size1=n_sites and size2=2*DIM. For mosaic, size1=n_sites and size2=top.
void print_lookup(long ** list, int size1, int size2){
	int i,j;
	for(i=0; i<size1; i++){
		printf("# Around site %d : \t",i);
		for(j=0; j<size2; j++) printf("%li\t", list[i][j]);
		printf("\n");
	}
}

// Prints out this whole code
void print_source(){
    printf("/* Source Code %s, created %s */\n",__FILE__,__DATE__);
    FILE *fp;
    int c;
   
    fp = fopen(__FILE__,"r");

    do {
     c = getc(fp); 
 	 putchar(c);
 	}
 	while(c != EOF);     
    fclose(fp);
}