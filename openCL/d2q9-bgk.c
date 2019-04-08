/*
  ------------------------------------------------
 |  Tesla K20m                                    |
  ------------------------------------------------
 |  MAX_COMPUTE_UNITS:         13                 |
 |  MAX_WORK_ITEM_DIMENSIONS:  3                  |
 |  MAX_WORK_ITEM_SIZES:       1024 / 1024 / 64   |
 |  MAX_WORK_GROUP_SIZE:       1024               |
 |  MAX_MEM_ALLOC_SIZE:        1199 MByte         |
 |  GLOBAL_MEM_SIZE:           4799 MByte         |
 |  ERROR_CORRECTION_SUPPORT:  yes                |
 |  LOCAL_MEM_TYPE:            local              |
 |  LOCAL_MEM_SIZE:            48 KByte           |
 |  MAX_CONSTANT_BUFFER_SIZE:  64 KByte           |
  ------------------------------------------------

  ------------------------------------------------
 |  Intel(R) Core(TM) i5-5257U CPU @ 2.70GHz      |
  ------------------------------------------------
 |  MAX_COMPUTE_UNITS:		    4                 |
 |  MAX_WORK_ITEM_DIMENSIONS:	3                 |
 |  MAX_WORK_ITEM_SIZES:		1024 / 1 / 1      |
 |  MAX_WORK_GROUP_SIZE:		1024              |
 |  MAX_MEM_ALLOC_SIZE:		    2048 MByte        |
 |  GLOBAL_MEM_SIZE:		    8192 MByte        |
 |  ERROR_CORRECTION_SUPPORT:	no                |
 |  LOCAL_MEM_TYPE:		        global            |
 |  LOCAL_MEM_SIZE:		        32 KByte          |
 |  MAX_CONSTANT_BUFFER_SIZE:	64 KByte          |
  ------------------------------------------------

  ------------------------------------------------
 |  Intel(R) Iris(TM) Graphics 6100               |
  ------------------------------------------------
 |  MAX_COMPUTE_UNITS:		    48                |
 |  MAX_WORK_ITEM_DIMENSIONS:	3                 |
 |  MAX_WORK_ITEM_SIZES:		256 / 256 / 256   |
 |  MAX_WORK_GROUP_SIZE:		256               |
 |  MAX_MEM_ALLOC_SIZE:		    384 MByte         |
 |  GLOBAL_MEM_SIZE:		    1536 MByte        |
 |  ERROR_CORRECTION_SUPPORT:	no                |
 |  LOCAL_MEM_TYPE:		        local             |
 |  LOCAL_MEM_SIZE:		        64 KByte          |
 |  MAX_CONSTANT_BUFFER_SIZE:	64 KByte          |
  ------------------------------------------------
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <sys/time.h>
#include <sys/resource.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"
#define OCLFILE         "kernels.cl"

#define LOCAL_NX 128
#define LOCAL_NY 1

/* struct to hold the parameter values */
typedef struct {
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
} t_param;

/* struct to hold OpenCL objects */
typedef struct {
  cl_device_id      device;
  cl_context        context;
  cl_command_queue  queue;

  cl_program program;
  cl_kernel  accelerate_flow;
  cl_kernel  prop_rebound_collision_avels;

  cl_mem cells_speed_0;
  cl_mem cells_speed_1;
  cl_mem cells_speed_2;
  cl_mem cells_speed_3;
  cl_mem cells_speed_4;
  cl_mem cells_speed_5;
  cl_mem cells_speed_6;
  cl_mem cells_speed_7;
  cl_mem cells_speed_8;
  cl_mem tmp_cells_speed_0;
  cl_mem tmp_cells_speed_1;
  cl_mem tmp_cells_speed_2;
  cl_mem tmp_cells_speed_3;
  cl_mem tmp_cells_speed_4;
  cl_mem tmp_cells_speed_5;
  cl_mem tmp_cells_speed_6;
  cl_mem tmp_cells_speed_7;
  cl_mem tmp_cells_speed_8;

  cl_mem obstacles;
  cl_mem av_vels;
  cl_mem partial_tot_u;
} t_ocl;

/* struct to hold the 'speed' values */
typedef struct {
  float* speed_0;
  float* speed_1;
  float* speed_2;
  float* speed_3;
  float* speed_4;
  float* speed_5;
  float* speed_6;
  float* speed_7;
  float* speed_8;
} t_soa;

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_soa** cells_ptr, t_soa** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, t_ocl* ocl);
/* precompute tot_cells */
int get_tot_cells(const t_param params, int* restrict obstacles);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
int timestep(const t_param params, const int tt, t_ocl ocl);
int write_values(const t_param params, t_soa* cells, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_soa** cells_ptr, t_soa** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr, t_ocl ocl);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_soa* cells);

/* compute average velocity */
float av_velocity_reynolds(const t_param params, t_soa* cells, int* obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_soa* cells, int* obstacles, t_ocl ocl);

/* utility functions */
void checkError(cl_int err, const char *op, const int line);
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

cl_device_id selectOpenCLDevice();

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[]) {

  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_ocl    ocl;                 /* struct to hold OpenCL objects */
  t_soa* cells     = NULL;    /* grid containing fluid densities */
  t_soa* tmp_cells = NULL;    /* scratch space */
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;      /* a record of the av. velocity computed for each timestep */
  cl_int err;
  struct timeval timstr;        /* structure to hold elapsed time */
  struct rusage ru;             /* structure to hold CPU time--system and user */
  double tic, toc;              /* floating point numbers to calculate elapsed wallclock time */
  double usrtim;                /* floating point number to record elapsed user CPU time */
  double systim;                /* floating point number to record elapsed system CPU time */

  /* parse the command line */
  if (argc != 3) {
    usage(argv[0]);
  } else {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* initialise our data structures and load values from file */
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels, &ocl);

  /* ----- iterate for maxIters timesteps ---------------------------------- */
  gettimeofday(&timstr, NULL);
  tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  // Write cells to OpenCL buffer
  err = clEnqueueWriteBuffer(
    ocl.queue, ocl.cells_speed_0, CL_TRUE, 0,
    sizeof(cl_float) * params.nx * params.ny, cells->speed_0, 0, NULL, NULL);
  checkError(err, "writing cells_speed_0 data", __LINE__);
  err = clEnqueueWriteBuffer(
    ocl.queue, ocl.cells_speed_1, CL_TRUE, 0,
    sizeof(cl_float) * params.nx * params.ny, cells->speed_1, 0, NULL, NULL);
  checkError(err, "writing cells_speed_1 data", __LINE__);
  err = clEnqueueWriteBuffer(
    ocl.queue, ocl.cells_speed_2, CL_TRUE, 0,
    sizeof(cl_float) * params.nx * params.ny, cells->speed_2, 0, NULL, NULL);
  checkError(err, "writing cells_speed_2 data", __LINE__);
  err = clEnqueueWriteBuffer(
    ocl.queue, ocl.cells_speed_3, CL_TRUE, 0,
    sizeof(cl_float) * params.nx * params.ny, cells->speed_3, 0, NULL, NULL);
  checkError(err, "writing cells_speed_3 data", __LINE__);
  err = clEnqueueWriteBuffer(
    ocl.queue, ocl.cells_speed_4, CL_TRUE, 0,
    sizeof(cl_float) * params.nx * params.ny, cells->speed_4, 0, NULL, NULL);
  checkError(err, "writing cells_speed_4 data", __LINE__);
  err = clEnqueueWriteBuffer(
    ocl.queue, ocl.cells_speed_5, CL_TRUE, 0,
    sizeof(cl_float) * params.nx * params.ny, cells->speed_5, 0, NULL, NULL);
  checkError(err, "writing cells_speed_5 data", __LINE__);
  err = clEnqueueWriteBuffer(
    ocl.queue, ocl.cells_speed_6, CL_TRUE, 0,
    sizeof(cl_float) * params.nx * params.ny, cells->speed_6, 0, NULL, NULL);
  checkError(err, "writing cells_speed_6 data", __LINE__);
  err = clEnqueueWriteBuffer(
    ocl.queue, ocl.cells_speed_7, CL_TRUE, 0,
    sizeof(cl_float) * params.nx * params.ny, cells->speed_7, 0, NULL, NULL);
  checkError(err, "writing cells_speed_7 data", __LINE__);
  err = clEnqueueWriteBuffer(
    ocl.queue, ocl.cells_speed_8, CL_TRUE, 0,
    sizeof(cl_float) * params.nx * params.ny, cells->speed_8, 0, NULL, NULL);
  checkError(err, "writing cells_speed_8 data", __LINE__);

  // Write obstacles to OpenCL buffer
  err = clEnqueueWriteBuffer(
    ocl.queue, ocl.obstacles, CL_TRUE, 0,
    sizeof(cl_int) * params.nx * params.ny, obstacles, 0, NULL, NULL);
  checkError(err, "writing obstacles data", __LINE__);

  for (int tt = 0; tt < params.maxIters; tt++) {
    timestep(params, tt, ocl);
#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells));
#endif
  }

  // Read cells from device
  err = clEnqueueReadBuffer(
    ocl.queue, ocl.cells_speed_0, CL_TRUE, 0,
    sizeof(cl_float) * params.nx * params.ny, cells->speed_0, 0, NULL, NULL);
  checkError(err, "reading cells_speed_0 data", __LINE__);
  err = clEnqueueReadBuffer(
    ocl.queue, ocl.cells_speed_1, CL_TRUE, 0,
    sizeof(cl_float) * params.nx * params.ny, cells->speed_1, 0, NULL, NULL);
  checkError(err, "reading cells_speed_1 data", __LINE__);
  err = clEnqueueReadBuffer(
    ocl.queue, ocl.cells_speed_2, CL_TRUE, 0,
    sizeof(cl_float) * params.nx * params.ny, cells->speed_2, 0, NULL, NULL);
  checkError(err, "reading cells_speed_2 data", __LINE__);
  err = clEnqueueReadBuffer(
    ocl.queue, ocl.cells_speed_3, CL_TRUE, 0,
    sizeof(cl_float) * params.nx * params.ny, cells->speed_3, 0, NULL, NULL);
  checkError(err, "reading cells_speed_3 data", __LINE__);
  err = clEnqueueReadBuffer(
    ocl.queue, ocl.cells_speed_4, CL_TRUE, 0,
    sizeof(cl_float) * params.nx * params.ny, cells->speed_4, 0, NULL, NULL);
  checkError(err, "reading cells_speed_4 data", __LINE__);
  err = clEnqueueReadBuffer(
    ocl.queue, ocl.cells_speed_5, CL_TRUE, 0,
    sizeof(cl_float) * params.nx * params.ny, cells->speed_5, 0, NULL, NULL);
  checkError(err, "reading cells_speed_5 data", __LINE__);
  err = clEnqueueReadBuffer(
    ocl.queue, ocl.cells_speed_6, CL_TRUE, 0,
    sizeof(cl_float) * params.nx * params.ny, cells->speed_6, 0, NULL, NULL);
  checkError(err, "reading cells_speed_6 data", __LINE__);
  err = clEnqueueReadBuffer(
    ocl.queue, ocl.cells_speed_7, CL_TRUE, 0,
    sizeof(cl_float) * params.nx * params.ny, cells->speed_7, 0, NULL, NULL);
  checkError(err, "reading cells_speed_7 data", __LINE__);
  err = clEnqueueReadBuffer(
    ocl.queue, ocl.cells_speed_8, CL_TRUE, 0,
    sizeof(cl_float) * params.nx * params.ny, cells->speed_8, 0, NULL, NULL);
  checkError(err, "reading cells_speed_8 data", __LINE__);

  // Read back partial sums
  int num_wrks = (params.nx / LOCAL_NX) * (params.ny / LOCAL_NY);
  float* h_partial_tot_u = malloc(sizeof(float) * num_wrks * params.maxIters);

  // Read partial_tot_u from device
  err = clEnqueueReadBuffer(ocl.queue, ocl.partial_tot_u, CL_TRUE, 0,
    sizeof(float) * num_wrks * params.maxIters, h_partial_tot_u, 0, NULL, NULL);
  checkError(err, "reading partial_tot_u data", __LINE__);

  // Compute average velocities at each step
  float tot_cells = (float) get_tot_cells(params, obstacles);
  for (int iter = 0; iter < params.maxIters; iter++) {
    float tot_u = 0.0f; // accumulated magnitudes of velocity for each cell
    for (int i = iter * num_wrks; i < (iter + 1) * num_wrks; i++) {
      tot_u += h_partial_tot_u[i];
    }
    av_vels[iter] = tot_u / tot_cells;
  }

  /* ----------------------------------------------------------------------- */
  gettimeofday(&timstr, NULL);
  toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  getrusage(RUSAGE_SELF, &ru);
  timstr = ru.ru_utime;
  usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  timstr = ru.ru_stime;
  systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  /* write final values and free memory */
  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles, ocl));
  printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
  printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
  printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
  write_values(params, cells, obstacles, av_vels);
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels, ocl);

  return EXIT_SUCCESS;
}

int timestep(const t_param params, const int tt, t_ocl ocl) {
    cl_int err;
    size_t global_2D[2] = {params.nx, params.ny};
    size_t global_1D[1] = {params.nx};
    size_t local[2]     = {LOCAL_NX, LOCAL_NY};

    /* --- Accelerate flow --- */
    // Set kernel arguments
    if (tt % 2 == 0) {
        err = clSetKernelArg(ocl.accelerate_flow, 0, sizeof(cl_mem), &ocl.cells_speed_0);
        checkError(err, "setting accelerate_flow arg 0 = cells_speed_0", __LINE__);
        err = clSetKernelArg(ocl.accelerate_flow, 1, sizeof(cl_mem), &ocl.cells_speed_1);
        checkError(err, "setting accelerate_flow arg 1 = cells_speed_1", __LINE__);
        err = clSetKernelArg(ocl.accelerate_flow, 2, sizeof(cl_mem), &ocl.cells_speed_2);
        checkError(err, "setting accelerate_flow arg 2 = cells_speed_2", __LINE__);
        err = clSetKernelArg(ocl.accelerate_flow, 3, sizeof(cl_mem), &ocl.cells_speed_3);
        checkError(err, "setting accelerate_flow arg 3 = cells_speed_3", __LINE__);
        err = clSetKernelArg(ocl.accelerate_flow, 4, sizeof(cl_mem), &ocl.cells_speed_4);
        checkError(err, "setting accelerate_flow arg 4 = cells_speed_4", __LINE__);
        err = clSetKernelArg(ocl.accelerate_flow, 5, sizeof(cl_mem), &ocl.cells_speed_5);
        checkError(err, "setting accelerate_flow arg 5 = cells_speed_5", __LINE__);
        err = clSetKernelArg(ocl.accelerate_flow, 6, sizeof(cl_mem), &ocl.cells_speed_6);
        checkError(err, "setting accelerate_flow arg 6 = cells_speed_6", __LINE__);
        err = clSetKernelArg(ocl.accelerate_flow, 7, sizeof(cl_mem), &ocl.cells_speed_7);
        checkError(err, "setting accelerate_flow arg 7 = cells_speed_7", __LINE__);
        err = clSetKernelArg(ocl.accelerate_flow, 8, sizeof(cl_mem), &ocl.cells_speed_8);
        checkError(err, "setting accelerate_flow arg 8 = cells_speed_8", __LINE__);
    } else {
        err = clSetKernelArg(ocl.accelerate_flow, 0, sizeof(cl_mem), &ocl.tmp_cells_speed_0);
        checkError(err, "setting accelerate_flow arg 0 = tmp_cells_speed_0", __LINE__);
        err = clSetKernelArg(ocl.accelerate_flow, 1, sizeof(cl_mem), &ocl.tmp_cells_speed_1);
        checkError(err, "setting accelerate_flow arg 1 = tmp_cells_speed_1", __LINE__);
        err = clSetKernelArg(ocl.accelerate_flow, 2, sizeof(cl_mem), &ocl.tmp_cells_speed_2);
        checkError(err, "setting accelerate_flow arg 2 = tmp_cells_speed_2", __LINE__);
        err = clSetKernelArg(ocl.accelerate_flow, 3, sizeof(cl_mem), &ocl.tmp_cells_speed_3);
        checkError(err, "setting accelerate_flow arg 3 = tmp_cells_speed_3", __LINE__);
        err = clSetKernelArg(ocl.accelerate_flow, 4, sizeof(cl_mem), &ocl.tmp_cells_speed_4);
        checkError(err, "setting accelerate_flow arg 4 = tmp_cells_speed_4", __LINE__);
        err = clSetKernelArg(ocl.accelerate_flow, 5, sizeof(cl_mem), &ocl.tmp_cells_speed_5);
        checkError(err, "setting accelerate_flow arg 5 = tmp_cells_speed_5", __LINE__);
        err = clSetKernelArg(ocl.accelerate_flow, 6, sizeof(cl_mem), &ocl.tmp_cells_speed_6);
        checkError(err, "setting accelerate_flow arg 6 = tmp_cells_speed_6", __LINE__);
        err = clSetKernelArg(ocl.accelerate_flow, 7, sizeof(cl_mem), &ocl.tmp_cells_speed_7);
        checkError(err, "setting accelerate_flow arg 7 = tmp_cells_speed_7", __LINE__);
        err = clSetKernelArg(ocl.accelerate_flow, 8, sizeof(cl_mem), &ocl.tmp_cells_speed_8);
        checkError(err, "setting accelerate_flow arg 8 = tmp_cells_speed_8", __LINE__);
    }
    err = clSetKernelArg(ocl.accelerate_flow, 9, sizeof(cl_mem), &ocl.obstacles);
    checkError(err, "setting accelerate_flow arg 9", __LINE__);
    err = clSetKernelArg(ocl.accelerate_flow, 10, sizeof(cl_int), &params.nx);
    checkError(err, "setting accelerate_flow arg 10", __LINE__);
    err = clSetKernelArg(ocl.accelerate_flow, 11, sizeof(cl_int), &params.ny);
    checkError(err, "setting accelerate_flow arg 11", __LINE__);
    err = clSetKernelArg(ocl.accelerate_flow, 12, sizeof(cl_float), &params.density);
    checkError(err, "setting accelerate_flow arg 12", __LINE__);
    err = clSetKernelArg(ocl.accelerate_flow, 13, sizeof(cl_float), &params.accel);
    checkError(err, "setting accelerate_flow arg 13", __LINE__);

    // Enqueue kernel
    err = clEnqueueNDRangeKernel(ocl.queue, ocl.accelerate_flow,
                                 1, NULL, global_1D, NULL, 0, NULL, NULL);
    checkError(err, "enqueueing accelerate_flow kernel", __LINE__);


    /* --- BIG KERNEL --- */
    int num_wrks = (global_2D[0] / local[0]) * (global_2D[1] / local[1]);
    int wrk_size = local[0] * local[1];

    if (tt % 2 == 0) {
        err = clSetKernelArg(ocl.prop_rebound_collision_avels, 0, sizeof(cl_mem), &ocl.cells_speed_0);
        checkError(err, "setting prop_rebound_collision_avels arg 0", __LINE__);
        err = clSetKernelArg(ocl.prop_rebound_collision_avels, 1, sizeof(cl_mem), &ocl.cells_speed_1);
        checkError(err, "setting prop_rebound_collision_avels arg 1", __LINE__);
        err = clSetKernelArg(ocl.prop_rebound_collision_avels, 2, sizeof(cl_mem), &ocl.cells_speed_2);
        checkError(err, "setting prop_rebound_collision_avels arg 2", __LINE__);
        err = clSetKernelArg(ocl.prop_rebound_collision_avels, 3, sizeof(cl_mem), &ocl.cells_speed_3);
        checkError(err, "setting prop_rebound_collision_avels arg 3", __LINE__);
        err = clSetKernelArg(ocl.prop_rebound_collision_avels, 4, sizeof(cl_mem), &ocl.cells_speed_4);
        checkError(err, "setting prop_rebound_collision_avels arg 4", __LINE__);
        err = clSetKernelArg(ocl.prop_rebound_collision_avels, 5, sizeof(cl_mem), &ocl.cells_speed_5);
        checkError(err, "setting prop_rebound_collision_avels arg 5", __LINE__);
        err = clSetKernelArg(ocl.prop_rebound_collision_avels, 6, sizeof(cl_mem), &ocl.cells_speed_6);
        checkError(err, "setting prop_rebound_collision_avels arg 6", __LINE__);
        err = clSetKernelArg(ocl.prop_rebound_collision_avels, 7, sizeof(cl_mem), &ocl.cells_speed_7);
        checkError(err, "setting prop_rebound_collision_avels arg 7", __LINE__);
        err = clSetKernelArg(ocl.prop_rebound_collision_avels, 8, sizeof(cl_mem), &ocl.cells_speed_8);
        checkError(err, "setting prop_rebound_collision_avels arg 8", __LINE__);

        err = clSetKernelArg(ocl.prop_rebound_collision_avels, 9, sizeof(cl_mem), &ocl.tmp_cells_speed_0);
        checkError(err, "setting prop_rebound_collision_avels arg 9", __LINE__);
        err = clSetKernelArg(ocl.prop_rebound_collision_avels, 10, sizeof(cl_mem), &ocl.tmp_cells_speed_1);
        checkError(err, "setting prop_rebound_collision_avels arg 10", __LINE__);
        err = clSetKernelArg(ocl.prop_rebound_collision_avels, 11, sizeof(cl_mem), &ocl.tmp_cells_speed_2);
        checkError(err, "setting prop_rebound_collision_avels arg 11", __LINE__);
        err = clSetKernelArg(ocl.prop_rebound_collision_avels, 12, sizeof(cl_mem), &ocl.tmp_cells_speed_3);
        checkError(err, "setting prop_rebound_collision_avels arg 12", __LINE__);
        err = clSetKernelArg(ocl.prop_rebound_collision_avels, 13, sizeof(cl_mem), &ocl.tmp_cells_speed_4);
        checkError(err, "setting prop_rebound_collision_avels arg 13", __LINE__);
        err = clSetKernelArg(ocl.prop_rebound_collision_avels, 14, sizeof(cl_mem), &ocl.tmp_cells_speed_5);
        checkError(err, "setting prop_rebound_collision_avels arg 14", __LINE__);
        err = clSetKernelArg(ocl.prop_rebound_collision_avels, 15, sizeof(cl_mem), &ocl.tmp_cells_speed_6);
        checkError(err, "setting prop_rebound_collision_avels arg 15", __LINE__);
        err = clSetKernelArg(ocl.prop_rebound_collision_avels, 16, sizeof(cl_mem), &ocl.tmp_cells_speed_7);
        checkError(err, "setting prop_rebound_collision_avels arg 16", __LINE__);
        err = clSetKernelArg(ocl.prop_rebound_collision_avels, 17, sizeof(cl_mem), &ocl.tmp_cells_speed_8);
        checkError(err, "setting prop_rebound_collision_avels arg 17", __LINE__);

    } else {
        err = clSetKernelArg(ocl.prop_rebound_collision_avels, 0, sizeof(cl_mem), &ocl.tmp_cells_speed_0);
        checkError(err, "setting prop_rebound_collision_avels arg 0", __LINE__);
        err = clSetKernelArg(ocl.prop_rebound_collision_avels, 1, sizeof(cl_mem), &ocl.tmp_cells_speed_1);
        checkError(err, "setting prop_rebound_collision_avels arg 1", __LINE__);
        err = clSetKernelArg(ocl.prop_rebound_collision_avels, 2, sizeof(cl_mem), &ocl.tmp_cells_speed_2);
        checkError(err, "setting prop_rebound_collision_avels arg 2", __LINE__);
        err = clSetKernelArg(ocl.prop_rebound_collision_avels, 3, sizeof(cl_mem), &ocl.tmp_cells_speed_3);
        checkError(err, "setting prop_rebound_collision_avels arg 3", __LINE__);
        err = clSetKernelArg(ocl.prop_rebound_collision_avels, 4, sizeof(cl_mem), &ocl.tmp_cells_speed_4);
        checkError(err, "setting prop_rebound_collision_avels arg 4", __LINE__);
        err = clSetKernelArg(ocl.prop_rebound_collision_avels, 5, sizeof(cl_mem), &ocl.tmp_cells_speed_5);
        checkError(err, "setting prop_rebound_collision_avels arg 5", __LINE__);
        err = clSetKernelArg(ocl.prop_rebound_collision_avels, 6, sizeof(cl_mem), &ocl.tmp_cells_speed_6);
        checkError(err, "setting prop_rebound_collision_avels arg 6", __LINE__);
        err = clSetKernelArg(ocl.prop_rebound_collision_avels, 7, sizeof(cl_mem), &ocl.tmp_cells_speed_7);
        checkError(err, "setting prop_rebound_collision_avels arg 7", __LINE__);
        err = clSetKernelArg(ocl.prop_rebound_collision_avels, 8, sizeof(cl_mem), &ocl.tmp_cells_speed_8);
        checkError(err, "setting prop_rebound_collision_avels arg 8", __LINE__);

        err = clSetKernelArg(ocl.prop_rebound_collision_avels, 9, sizeof(cl_mem), &ocl.cells_speed_0);
        checkError(err, "setting prop_rebound_collision_avels arg 9", __LINE__);
        err = clSetKernelArg(ocl.prop_rebound_collision_avels, 10, sizeof(cl_mem), &ocl.cells_speed_1);
        checkError(err, "setting prop_rebound_collision_avels arg 10", __LINE__);
        err = clSetKernelArg(ocl.prop_rebound_collision_avels, 11, sizeof(cl_mem), &ocl.cells_speed_2);
        checkError(err, "setting prop_rebound_collision_avels arg 11", __LINE__);
        err = clSetKernelArg(ocl.prop_rebound_collision_avels, 12, sizeof(cl_mem), &ocl.cells_speed_3);
        checkError(err, "setting prop_rebound_collision_avels arg 12", __LINE__);
        err = clSetKernelArg(ocl.prop_rebound_collision_avels, 13, sizeof(cl_mem), &ocl.cells_speed_4);
        checkError(err, "setting prop_rebound_collision_avels arg 13", __LINE__);
        err = clSetKernelArg(ocl.prop_rebound_collision_avels, 14, sizeof(cl_mem), &ocl.cells_speed_5);
        checkError(err, "setting prop_rebound_collision_avels arg 14", __LINE__);
        err = clSetKernelArg(ocl.prop_rebound_collision_avels, 15, sizeof(cl_mem), &ocl.cells_speed_6);
        checkError(err, "setting prop_rebound_collision_avels arg 15", __LINE__);
        err = clSetKernelArg(ocl.prop_rebound_collision_avels, 16, sizeof(cl_mem), &ocl.cells_speed_7);
        checkError(err, "setting prop_rebound_collision_avels arg 16", __LINE__);
        err = clSetKernelArg(ocl.prop_rebound_collision_avels, 17, sizeof(cl_mem), &ocl.cells_speed_8);
        checkError(err, "setting prop_rebound_collision_avels arg 17", __LINE__);
    }
    err = clSetKernelArg(ocl.prop_rebound_collision_avels, 18, sizeof(cl_mem), &ocl.obstacles);
    checkError(err, "setting prop_rebound_collision_avels arg 18", __LINE__);
    err = clSetKernelArg(ocl.prop_rebound_collision_avels, 19, sizeof(cl_int), &params.nx);
    checkError(err, "setting prop_rebound_collision_avels arg 19", __LINE__);
    err = clSetKernelArg(ocl.prop_rebound_collision_avels, 20, sizeof(cl_int), &params.ny);
    checkError(err, "setting prop_rebound_collision_avels arg 20", __LINE__);
    err = clSetKernelArg(ocl.prop_rebound_collision_avels, 21, sizeof(cl_int), &tt);
    checkError(err, "setting prop_rebound_collision_avels arg 21", __LINE__);
    err = clSetKernelArg(ocl.prop_rebound_collision_avels, 22, sizeof(cl_float), &params.omega);
    checkError(err, "setting prop_rebound_collision_avels arg 22", __LINE__);
    err = clSetKernelArg(ocl.prop_rebound_collision_avels, 23, sizeof(cl_mem), &ocl.partial_tot_u);
    checkError(err, "setting prop_rebound_collision_avels arg 23", __LINE__);
    err = clSetKernelArg(ocl.prop_rebound_collision_avels, 24, sizeof(cl_float) * wrk_size, NULL);
    checkError(err, "setting prop_rebound_collision_avels arg 24", __LINE__);

    // Enqueue kernel
    err = clEnqueueNDRangeKernel(ocl.queue, ocl.prop_rebound_collision_avels,
                                 2, NULL, global_2D, local, 0, NULL, NULL);
    checkError(err, "enqueueing prop_rebound_collision_avels kernel", __LINE__);


    return EXIT_SUCCESS;
}

float av_velocity_reynolds(const t_param params, t_soa* cells, int* obstacles) {
    int    tot_cells = 0;  /* no. of cells used in calculation */
    float tot_u;          /* accumulated magnitudes of velocity for each cell */

    /* initialise */
    tot_u = 0.f;

    /* loop over all non-blocked cells */
    for (int jj = 0; jj < params.ny; jj++) {
      for (int ii = 0; ii < params.nx; ii++) {
        /* ignore occupied cells */
        if (!obstacles[ii + jj*params.nx]) {
          /* local density total */
         float local_density =  cells->speed_0[ii + jj*params.nx] +
                                cells->speed_1[ii + jj*params.nx] +
                                cells->speed_2[ii + jj*params.nx] +
                                cells->speed_3[ii + jj*params.nx] +
                                cells->speed_4[ii + jj*params.nx] +
                                cells->speed_5[ii + jj*params.nx] +
                                cells->speed_6[ii + jj*params.nx] +
                                cells->speed_7[ii + jj*params.nx] +
                                cells->speed_8[ii + jj*params.nx];

          /* x-component of velocity */
          float u_x = (cells->speed_1[ii + jj*params.nx]
                    +  cells->speed_5[ii + jj*params.nx]
                    +  cells->speed_8[ii + jj*params.nx]
                    - (cells->speed_3[ii + jj*params.nx]
                    +  cells->speed_6[ii + jj*params.nx]
                    +  cells->speed_7[ii + jj*params.nx]))
                    / local_density;
          /* compute y velocity component */
          float u_y = (cells->speed_2[ii + jj*params.nx]
                    +  cells->speed_5[ii + jj*params.nx]
                    +  cells->speed_6[ii + jj*params.nx]
                    - (cells->speed_4[ii + jj*params.nx]
                    +  cells->speed_7[ii + jj*params.nx]
                    +  cells->speed_8[ii + jj*params.nx]))
                    / local_density;
          /* accumulate the norm of x- and y- velocity components */
          tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
          /* increase counter of inspected cells */
          tot_cells++;
        }
      }
    }

    return tot_u / (float)tot_cells;
}

int get_tot_cells(const t_param params, int* restrict obstacles) {
    int tot_cells  = 0;
    for (int jj = 0; jj < params.ny; jj++) {
      for (int ii = 0; ii < params.nx; ii++) {
        tot_cells += (obstacles[jj*params.nx + ii] != 0) ? 0 : 1;
      }
    }
    return tot_cells;
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_soa** cells_ptr, t_soa** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, t_ocl *ocl) {
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */
  char*  ocl_src;        /* OpenCL kernel source */
  long   ocl_size;       /* size of OpenCL kernel source */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL) {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */
  *cells_ptr = (t_soa*) malloc(sizeof(t_soa));

  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  (*cells_ptr)->speed_0 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*cells_ptr)->speed_1 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*cells_ptr)->speed_2 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*cells_ptr)->speed_3 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*cells_ptr)->speed_4 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*cells_ptr)->speed_5 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*cells_ptr)->speed_6 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*cells_ptr)->speed_7 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*cells_ptr)->speed_8 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);

  /* 'helper' grid, used as scratch space */
  *tmp_cells_ptr = (t_soa*) malloc(sizeof(t_soa));

  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  (*tmp_cells_ptr)->speed_0 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*tmp_cells_ptr)->speed_1 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*tmp_cells_ptr)->speed_2 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*tmp_cells_ptr)->speed_3 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*tmp_cells_ptr)->speed_4 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*tmp_cells_ptr)->speed_5 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*tmp_cells_ptr)->speed_6 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*tmp_cells_ptr)->speed_7 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*tmp_cells_ptr)->speed_8 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  for (int jj = 0; jj < params->ny; jj++) {
    for (int ii = 0; ii < params->nx; ii++) {
      /* centre */
      (*cells_ptr)->speed_0[ii + jj*params->nx] = w0;
      /* axis directions */
      (*cells_ptr)->speed_1[ii + jj*params->nx] = w1;
      (*cells_ptr)->speed_2[ii + jj*params->nx] = w1;
      (*cells_ptr)->speed_3[ii + jj*params->nx] = w1;
      (*cells_ptr)->speed_4[ii + jj*params->nx] = w1;
      /* diagonals */
      (*cells_ptr)->speed_5[ii + jj*params->nx] = w2;
      (*cells_ptr)->speed_6[ii + jj*params->nx] = w2;
      (*cells_ptr)->speed_7[ii + jj*params->nx] = w2;
      (*cells_ptr)->speed_8[ii + jj*params->nx] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj < params->ny; jj++) {
    for (int ii = 0; ii < params->nx; ii++) {
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL) {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF) {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[xx + yy*params->nx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  /* --- OpenCL initialisation section --- */
  cl_int err;

  ocl->device = selectOpenCLDevice();

  // Create OpenCL context
  ocl->context = clCreateContext(NULL, 1, &ocl->device, NULL, NULL, &err);
  checkError(err, "creating context", __LINE__);

  fp = fopen(OCLFILE, "r");
  if (fp == NULL) {
    sprintf(message, "could not open OpenCL kernel file: %s", OCLFILE);
    die(message, __LINE__, __FILE__);
  }

  // Create OpenCL command queue
  ocl->queue = clCreateCommandQueue(ocl->context, ocl->device, 0, &err);
  checkError(err, "creating command queue", __LINE__);

  // Load OpenCL kernel source
  fseek(fp, 0, SEEK_END);
  ocl_size = ftell(fp) + 1;
  ocl_src = (char*)malloc(ocl_size);
  memset(ocl_src, 0, ocl_size);
  fseek(fp, 0, SEEK_SET);
  fread(ocl_src, 1, ocl_size, fp);
  fclose(fp);

  // Create OpenCL program
  ocl->program = clCreateProgramWithSource(
    ocl->context, 1, (const char**)&ocl_src, NULL, &err);
  free(ocl_src);
  checkError(err, "creating program", __LINE__);

  // Build OpenCL program
  char flags[] = "-cl-denorms-are-zero -cl-strict-aliasing -cl-fast-relaxed-math";
   err = clBuildProgram(ocl->program, 1, &ocl->device, flags, NULL, NULL);
  if (err == CL_BUILD_PROGRAM_FAILURE) {
    size_t sz;
    clGetProgramBuildInfo(
      ocl->program, ocl->device,
      CL_PROGRAM_BUILD_LOG, 0, NULL, &sz);
    char *buildlog = malloc(sz);
    clGetProgramBuildInfo(
      ocl->program, ocl->device,
      CL_PROGRAM_BUILD_LOG, sz, buildlog, NULL);
    fprintf(stderr, "\nOpenCL build log:\n\n%s\n", buildlog);
    free(buildlog);
  }
  checkError(err, "building program", __LINE__);

  // Create OpenCL kernels
  ocl->accelerate_flow = clCreateKernel(ocl->program, "accelerate_flow", &err);
  checkError(err, "creating accelerate_flow kernel", __LINE__);
  ocl->prop_rebound_collision_avels = clCreateKernel(ocl->program, "prop_rebound_collision_avels", &err);
  checkError(err, "creating prop_rebound_collision_avels kernel", __LINE__);

  // Allocate OpenCL buffers
  ocl->cells_speed_0 = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(cl_float) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating cells_speed_0 buffer", __LINE__);
  ocl->cells_speed_1 = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(cl_float) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating cells_speed_1 buffer", __LINE__);
  ocl->cells_speed_2 = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(cl_float) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating cells_speed_2 buffer", __LINE__);
  ocl->cells_speed_3 = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(cl_float) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating cells_speed_3 buffer", __LINE__);
  ocl->cells_speed_4 = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(cl_float) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating cells_speed_4 buffer", __LINE__);
  ocl->cells_speed_5 = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(cl_float) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating cells_speed_5 buffer", __LINE__);
  ocl->cells_speed_6 = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(cl_float) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating cells_speed_6 buffer", __LINE__);
  ocl->cells_speed_7 = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(cl_float) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating cells_speed_7 buffer", __LINE__);
  ocl->cells_speed_8 = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(cl_float) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating cells_speed_8 buffer", __LINE__);

  ocl->tmp_cells_speed_0 = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(cl_float) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating tmp_cells_speed_0 buffer", __LINE__);
  ocl->tmp_cells_speed_1 = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(cl_float) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating tmp_cells_speed_1 buffer", __LINE__);
  ocl->tmp_cells_speed_2 = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(cl_float) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating tmp_cells_speed_2 buffer", __LINE__);
  ocl->tmp_cells_speed_3 = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(cl_float) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating tmp_cells_speed_3 buffer", __LINE__);
  ocl->tmp_cells_speed_4 = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(cl_float) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating tmp_cells_speed_4 buffer", __LINE__);
  ocl->tmp_cells_speed_5 = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(cl_float) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating tmp_cells_speed_5 buffer", __LINE__);
  ocl->tmp_cells_speed_6 = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(cl_float) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating tmp_cells_speed_6 buffer", __LINE__);
  ocl->tmp_cells_speed_7 = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(cl_float) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating tmp_cells_speed_7 buffer", __LINE__);
  ocl->tmp_cells_speed_8 = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(cl_float) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating tmp_cells_speed_8 buffer", __LINE__);

  ocl->obstacles = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(cl_int) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating obstacles buffer", __LINE__);
  ocl->av_vels = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(cl_float) * params->maxIters, NULL, &err);
  checkError(err, "creating av_vels buffer", __LINE__);
  int num_wrks = (params->nx / LOCAL_NX) * (params->ny / LOCAL_NY);
  ocl->partial_tot_u = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(cl_float) * num_wrks * params->maxIters, NULL, &err);
  checkError(err, "creating partial_tot_u buffer", __LINE__);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_soa** cells_ptr, t_soa** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr, t_ocl ocl) {
  /*
  ** free up allocated memory
  */
  // free(*cells_ptr);
  // *cells_ptr = NULL;
  //
  // free(*tmp_cells_ptr);
  // *tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  clReleaseMemObject(ocl.cells_speed_0);
  clReleaseMemObject(ocl.cells_speed_1);
  clReleaseMemObject(ocl.cells_speed_2);
  clReleaseMemObject(ocl.cells_speed_3);
  clReleaseMemObject(ocl.cells_speed_4);
  clReleaseMemObject(ocl.cells_speed_5);
  clReleaseMemObject(ocl.cells_speed_6);
  clReleaseMemObject(ocl.cells_speed_7);
  clReleaseMemObject(ocl.cells_speed_8);
  clReleaseMemObject(ocl.tmp_cells_speed_0);
  clReleaseMemObject(ocl.tmp_cells_speed_1);
  clReleaseMemObject(ocl.tmp_cells_speed_2);
  clReleaseMemObject(ocl.tmp_cells_speed_3);
  clReleaseMemObject(ocl.tmp_cells_speed_4);
  clReleaseMemObject(ocl.tmp_cells_speed_5);
  clReleaseMemObject(ocl.tmp_cells_speed_6);
  clReleaseMemObject(ocl.tmp_cells_speed_7);
  clReleaseMemObject(ocl.tmp_cells_speed_8);

  clReleaseMemObject(ocl.obstacles);
  clReleaseKernel(ocl.accelerate_flow);
  clReleaseKernel(ocl.prop_rebound_collision_avels);
  clReleaseProgram(ocl.program);
  clReleaseCommandQueue(ocl.queue);
  clReleaseContext(ocl.context);

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, t_soa* cells, int* obstacles, t_ocl ocl) {
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity_reynolds(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_soa* cells) {
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < params.ny; jj++) {
    for (int ii = 0; ii < params.nx; ii++) {
      total += cells->speed_0[ii + jj*params.nx] +
               cells->speed_1[ii + jj*params.nx] +
               cells->speed_2[ii + jj*params.nx] +
               cells->speed_3[ii + jj*params.nx] +
               cells->speed_4[ii + jj*params.nx] +
               cells->speed_5[ii + jj*params.nx] +
               cells->speed_6[ii + jj*params.nx] +
               cells->speed_7[ii + jj*params.nx] +
               cells->speed_8[ii + jj*params.nx];
    }
  }

  return total;
}

int write_values(const t_param params, t_soa* cells, int* obstacles, float* av_vels) {
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL) {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; jj++)  {
    for (int ii = 0; ii < params.nx; ii++) {
      /* an occupied cell */
      if (obstacles[ii + jj*params.nx]) {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else {
        local_density = cells->speed_0[ii + jj*params.nx] +
                        cells->speed_1[ii + jj*params.nx] +
                        cells->speed_2[ii + jj*params.nx] +
                        cells->speed_3[ii + jj*params.nx] +
                        cells->speed_4[ii + jj*params.nx] +
                        cells->speed_5[ii + jj*params.nx] +
                        cells->speed_6[ii + jj*params.nx] +
                        cells->speed_7[ii + jj*params.nx] +
                        cells->speed_8[ii + jj*params.nx];

        /* compute x velocity component */
        u_x = (cells->speed_1[ii + jj*params.nx]
            +  cells->speed_5[ii + jj*params.nx]
            +  cells->speed_8[ii + jj*params.nx]
            - (cells->speed_3[ii + jj*params.nx]
            +  cells->speed_6[ii + jj*params.nx]
            +  cells->speed_7[ii + jj*params.nx]))
            / local_density;
        /* compute y velocity component */
        u_y = (cells->speed_2[ii + jj*params.nx]
            +  cells->speed_5[ii + jj*params.nx]
            +  cells->speed_6[ii + jj*params.nx]
            - (cells->speed_4[ii + jj*params.nx]
            +  cells->speed_7[ii + jj*params.nx]
            +  cells->speed_8[ii + jj*params.nx]))
            / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL) {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++) {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void checkError(cl_int err, const char *op, const int line) {
  if (err != CL_SUCCESS) {
    fprintf(stderr, "OpenCL error during '%s' on line %d: %d\n", op, line, err);
    fflush(stderr);
    exit(EXIT_FAILURE);
  }
}

void die(const char* message, const int line, const char* file) {
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe) {
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}

#define MAX_DEVICES 32
#define MAX_DEVICE_NAME 1024

cl_device_id selectOpenCLDevice() {
  cl_int err;
  cl_uint num_platforms = 0;
  cl_uint total_devices = 0;
  cl_platform_id platforms[8];
  cl_device_id devices[MAX_DEVICES];
  char name[MAX_DEVICE_NAME];

  // Get list of platforms
  err = clGetPlatformIDs(8, platforms, &num_platforms);
  checkError(err, "getting platforms", __LINE__);

  // Get list of devices
  for (cl_uint p = 0; p < num_platforms; p++) {
    cl_uint num_devices = 0;
    err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL,
                         MAX_DEVICES-total_devices, devices+total_devices,
                         &num_devices);
    checkError(err, "getting device name", __LINE__);
    total_devices += num_devices;
  }
  /*
  // Print list of devices
  printf("\nAvailable OpenCL devices:\n");
  for (cl_uint d = 0; d < total_devices; d++) {
    clGetDeviceInfo(devices[d], CL_DEVICE_NAME, MAX_DEVICE_NAME, name, NULL);
    printf("%2d: %s\n", d, name);
  }
  printf("\n");
  */

  // Use first device unless OCL_DEVICE environment variable used
  cl_uint device_index = 0;
  char *dev_env = getenv("OCL_DEVICE");
  if (dev_env) {
    char *end;
    device_index = strtol(dev_env, &end, 10);
    if (strlen(end))
      die("invalid OCL_DEVICE variable", __LINE__, __FILE__);
  }

  if (device_index >= total_devices) {
    fprintf(stderr, "device index set to %d but only %d devices available\n",
            device_index, total_devices);
    exit(1);
  }

  // Print OpenCL device name
  clGetDeviceInfo(devices[device_index], CL_DEVICE_NAME,
                  MAX_DEVICE_NAME, name, NULL);
  printf("Selected OpenCL device:\n-> %s (index=%d)\n\n", name, device_index);

  return devices[device_index];
}
