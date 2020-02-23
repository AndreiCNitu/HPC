#include <iostream>
#include <CL/sycl.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>

class acc_flow_1;
class acc_flow_2;
class lattice_boltzmann_1;
class lattice_boltzmann_2;
class reduction_1;
class reduction_2;

using accessor_t =
  cl::sycl::accessor<float, 1, 
                     cl::sycl::access::mode::read_write, 
                     cl::sycl::access::target::global_buffer>;
using read_accessor_t =
  cl::sycl::accessor<float, 1, 
                     cl::sycl::access::mode::read, 
                     cl::sycl::access::target::global_buffer>;
using write_accessor_t =
  cl::sycl::accessor<float, 1, 
                     cl::sycl::access::mode::write, 
                     cl::sycl::access::target::global_buffer>;
using iread_accessor_t =
  cl::sycl::accessor<int, 1, 
                     cl::sycl::access::mode::read, 
                     cl::sycl::access::target::global_buffer>;
using local_accessor_t =
  cl::sycl::accessor<float, 1, 
                     cl::sycl::access::mode::read_write,
                     cl::sycl::access::target::local>;

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

// Define output file name
#define OUTPUT_FILE "stencil.pgm"

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

__host__ __device__ inline void accelerate_flow(
    accessor_t speeds_0_acc,
    accessor_t speeds_1_acc,
    accessor_t speeds_2_acc,
    accessor_t speeds_3_acc,
    accessor_t speeds_4_acc,
    accessor_t speeds_5_acc,
    accessor_t speeds_6_acc,
    accessor_t speeds_7_acc,
    accessor_t speeds_8_acc,
    iread_accessor_t obstacles_acc,
    cl::sycl::item<1> item,
    const t_param params);

__host__ __device__ inline void lbm_computation(
    accessor_t speeds_0_acc,
    accessor_t speeds_1_acc,
    accessor_t speeds_2_acc,
    accessor_t speeds_3_acc,
    accessor_t speeds_4_acc,
    accessor_t speeds_5_acc,
    accessor_t speeds_6_acc,
    accessor_t speeds_7_acc,
    accessor_t speeds_8_acc,
    accessor_t tmp_speeds_0_acc,   
    accessor_t tmp_speeds_1_acc,
    accessor_t tmp_speeds_2_acc,
    accessor_t tmp_speeds_3_acc,
    accessor_t tmp_speeds_4_acc,
    accessor_t tmp_speeds_5_acc,
    accessor_t tmp_speeds_6_acc,
    accessor_t tmp_speeds_7_acc,
    accessor_t tmp_speeds_8_acc,
    iread_accessor_t obstacles_acc,
    accessor_t tot_u_acc,
    accessor_t av_vels_acc, 
    cl::sycl::item<2> item,
    const t_param params);

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_soa** cells_ptr, t_soa** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
int timestep(const t_param params, t_soa* cells, t_soa* tmp_cells, int* obstacles, float* av_vels, const int tt);
int accelerate_flow(const t_param params, t_soa* cells, int* obstacles);
float prop_rebound_collision_avels(const t_param params, t_soa* cells, t_soa* tmp_cells, int* obstacles);
int write_values(const t_param params, t_soa* cells, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_soa** cells_ptr, t_soa** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_soa* cells);

/* compute average velocity */
float av_velocity(const t_param params, t_soa* cells, int* obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_soa* cells, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

int main(int argc, char* argv[]) {
  
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_soa* cells     = NULL;    /* grid containing fluid densities */
  t_soa* tmp_cells = NULL;    /* scratch space */
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;      /* a record of the av. velocity computed for each timestep */
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
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels);

  int tot_cells  = 0;
  for (int jj = 0; jj < params.ny; jj++) {
    for (int ii = 0; ii < params.nx; ii++) {
      tot_cells += (obstacles[jj*params.nx + ii] != 0) ? 0 : 1;
    }
  }

  cl::sycl::default_selector device_selector;

  cl::sycl::queue queue(device_selector);
  std::cout << "Running on "
            << queue.get_device().get_info<cl::sycl::info::device::name>()
            << "\n";

  /* iterate for maxIters timesteps */
  gettimeofday(&timstr, NULL);
  tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  {
    const int lx = params.nx;
    const int ly = params.ny;

    cl::sycl::buffer<float, 1> tot_u_sycl(cl::sycl::range<1>(lx * ly));
    cl::sycl::buffer<float, 1> av_vels_sycl(av_vels, cl::sycl::range<1>(params.maxIters));

    cl::sycl::buffer<int, 1> obstacles_sycl(obstacles, cl::sycl::range<1>(lx * ly));

    cl::sycl::buffer<float, 1> speeds_0_sycl(cells->speed_0, cl::sycl::range<1>(lx * ly));
    cl::sycl::buffer<float, 1> speeds_1_sycl(cells->speed_1, cl::sycl::range<1>(lx * ly));
    cl::sycl::buffer<float, 1> speeds_2_sycl(cells->speed_2, cl::sycl::range<1>(lx * ly));
    cl::sycl::buffer<float, 1> speeds_3_sycl(cells->speed_3, cl::sycl::range<1>(lx * ly));
    cl::sycl::buffer<float, 1> speeds_4_sycl(cells->speed_4, cl::sycl::range<1>(lx * ly));
    cl::sycl::buffer<float, 1> speeds_5_sycl(cells->speed_5, cl::sycl::range<1>(lx * ly));
    cl::sycl::buffer<float, 1> speeds_6_sycl(cells->speed_6, cl::sycl::range<1>(lx * ly));
    cl::sycl::buffer<float, 1> speeds_7_sycl(cells->speed_7, cl::sycl::range<1>(lx * ly));
    cl::sycl::buffer<float, 1> speeds_8_sycl(cells->speed_8, cl::sycl::range<1>(lx * ly));
    
    cl::sycl::buffer<float, 1> tmp_speeds_0_sycl(tmp_cells->speed_0, cl::sycl::range<1>(lx * ly));
    cl::sycl::buffer<float, 1> tmp_speeds_1_sycl(tmp_cells->speed_1, cl::sycl::range<1>(lx * ly));
    cl::sycl::buffer<float, 1> tmp_speeds_2_sycl(tmp_cells->speed_2, cl::sycl::range<1>(lx * ly));
    cl::sycl::buffer<float, 1> tmp_speeds_3_sycl(tmp_cells->speed_3, cl::sycl::range<1>(lx * ly));
    cl::sycl::buffer<float, 1> tmp_speeds_4_sycl(tmp_cells->speed_4, cl::sycl::range<1>(lx * ly));
    cl::sycl::buffer<float, 1> tmp_speeds_5_sycl(tmp_cells->speed_5, cl::sycl::range<1>(lx * ly));
    cl::sycl::buffer<float, 1> tmp_speeds_6_sycl(tmp_cells->speed_6, cl::sycl::range<1>(lx * ly));
    cl::sycl::buffer<float, 1> tmp_speeds_7_sycl(tmp_cells->speed_7, cl::sycl::range<1>(lx * ly));
    cl::sycl::buffer<float, 1> tmp_speeds_8_sycl(tmp_cells->speed_8, cl::sycl::range<1>(lx * ly));

    for(int iter = 0; iter < params.maxIters / 2; iter++) {
      queue.submit([&] (cl::sycl::handler& cgh) {
   
        auto tot_u_acc = tot_u_sycl.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto av_vels_acc = av_vels_sycl.get_access<cl::sycl::access::mode::read_write>(cgh);

        auto obstacles_acc = obstacles_sycl.get_access<cl::sycl::access::mode::read>(cgh);

        auto speeds_0_acc = speeds_0_sycl.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto speeds_1_acc = speeds_1_sycl.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto speeds_2_acc = speeds_2_sycl.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto speeds_3_acc = speeds_3_sycl.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto speeds_4_acc = speeds_4_sycl.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto speeds_5_acc = speeds_5_sycl.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto speeds_6_acc = speeds_6_sycl.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto speeds_7_acc = speeds_7_sycl.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto speeds_8_acc = speeds_8_sycl.get_access<cl::sycl::access::mode::read_write>(cgh);
        
        auto tmp_speeds_0_acc = tmp_speeds_0_sycl.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto tmp_speeds_1_acc = tmp_speeds_1_sycl.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto tmp_speeds_2_acc = tmp_speeds_2_sycl.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto tmp_speeds_3_acc = tmp_speeds_3_sycl.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto tmp_speeds_4_acc = tmp_speeds_4_sycl.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto tmp_speeds_5_acc = tmp_speeds_5_sycl.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto tmp_speeds_6_acc = tmp_speeds_6_sycl.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto tmp_speeds_7_acc = tmp_speeds_7_sycl.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto tmp_speeds_8_acc = tmp_speeds_8_sycl.get_access<cl::sycl::access::mode::read_write>(cgh);
       
        cgh.parallel_for<class acc_flow_1>(cl::sycl::range<1>(lx), [=](cl::sycl::item<1> item) {
          
          accelerate_flow(speeds_0_acc, speeds_1_acc, speeds_2_acc,
                          speeds_3_acc, speeds_4_acc, speeds_5_acc,
                          speeds_6_acc, speeds_7_acc, speeds_8_acc,
                          obstacles_acc, item, params);
        });
        cgh.parallel_for<class lattice_boltzmann_1>(cl::sycl::range<2>(lx, ly), [=](cl::sycl::item<2> item) {

          lbm_computation(speeds_0_acc, speeds_1_acc, speeds_2_acc,
                          speeds_3_acc, speeds_4_acc, speeds_5_acc,
                          speeds_6_acc, speeds_7_acc, speeds_8_acc,
                          tmp_speeds_0_acc, tmp_speeds_1_acc, tmp_speeds_2_acc,
                          tmp_speeds_3_acc, tmp_speeds_4_acc, tmp_speeds_5_acc,
                          tmp_speeds_6_acc, tmp_speeds_7_acc, tmp_speeds_8_acc,
                          obstacles_acc, tot_u_acc, av_vels_acc, item, params);
        });
        cgh.single_task<class reduction_1>( [=]() {   
          float tot_u = 0.0f;
          for(int jj = 0; jj < params.ny; jj++) {
            for(int ii = 0; ii < params.nx; ii++) {
              tot_u += tot_u_acc[jj*params.nx + ii];
            }
          }
          av_vels_acc[2 * iter] = tot_u / (float) tot_cells;
        });    
        cgh.parallel_for<class acc_flow_2>(cl::sycl::range<1>(lx), [=](cl::sycl::item<1> item) {
        
          accelerate_flow(tmp_speeds_0_acc, tmp_speeds_1_acc, tmp_speeds_2_acc,
                          tmp_speeds_3_acc, tmp_speeds_4_acc, tmp_speeds_5_acc,
                          tmp_speeds_6_acc, tmp_speeds_7_acc, tmp_speeds_8_acc,
                          obstacles_acc, item, params);
        });
        cgh.parallel_for<class lattice_boltzmann_2>(cl::sycl::range<2>(lx, ly), [=](cl::sycl::item<2> item) {
        
          lbm_computation(tmp_speeds_0_acc, tmp_speeds_1_acc, tmp_speeds_2_acc,
                          tmp_speeds_3_acc, tmp_speeds_4_acc, tmp_speeds_5_acc,
                          tmp_speeds_6_acc, tmp_speeds_7_acc, tmp_speeds_8_acc,
                          speeds_0_acc, speeds_1_acc, speeds_2_acc,
                          speeds_3_acc, speeds_4_acc, speeds_5_acc,
                          speeds_6_acc, speeds_7_acc, speeds_8_acc,
                          obstacles_acc, tot_u_acc, av_vels_acc, item, params);
        });
        cgh.single_task<class reduction_2>( [=]() {   
          float tot_u = 0.0f;
          for(int jj = 0; jj < params.ny; jj++) {
            for(int ii = 0; ii < params.nx; ii++) {
              tot_u += tot_u_acc[jj*params.nx + ii];
            }
          }
          av_vels_acc[2 * iter + 1] = tot_u / (float) tot_cells;
        });    
      });
    }
  }/*
  for (int tt = 0; tt < params.maxIters; tt++) {
    timestep(params,     cells, tmp_cells, obstacles, av_vels, tt); tt++;
    timestep(params, tmp_cells,     cells, obstacles, av_vels, tt);
  }*/
  gettimeofday(&timstr, NULL);
  toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  getrusage(RUSAGE_SELF, &ru);
  timstr = ru.ru_utime;
  usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  timstr = ru.ru_stime;
  systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  /* write final values and free memory */
  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
  printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
  printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
  printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
  printf(" memory bandwidth: %lf GB/s\n", (4 * 18 * (double)(params.nx / 1024) * (double)(params.ny / 1024) * params.maxIters) / ((toc - tic) * 1024) );
  write_values(params, cells, obstacles, av_vels);
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);

  return EXIT_SUCCESS;
}

__host__ __device__ inline void accelerate_flow(
    accessor_t speeds_0_acc,
    accessor_t speeds_1_acc,
    accessor_t speeds_2_acc,
    accessor_t speeds_3_acc,
    accessor_t speeds_4_acc,
    accessor_t speeds_5_acc,
    accessor_t speeds_6_acc,
    accessor_t speeds_7_acc,
    accessor_t speeds_8_acc,
    iread_accessor_t obstacles_acc,
    cl::sycl::item<1> item,
    const t_param params)
{
  /* modify the 2nd row of the grid */
  const int ii = item[0];
  const int jj = params.ny - 2;

  /* compute weighting factors */
  const float w1 = params.density * params.accel / 9.f;
  const float w2 = params.density * params.accel / 36.f;

  /* if the cell is not occupied and
  ** we don't send a negative density */
  if (!obstacles_acc[ii + jj * params.nx]
      && (speeds_3_acc[ii + jj * params.nx] - w1) > 0.f
      && (speeds_6_acc[ii + jj * params.nx] - w2) > 0.f
      && (speeds_7_acc[ii + jj * params.nx] - w2) > 0.f) {
    /* increase 'east-side' densities */
    speeds_1_acc[ii + jj * params.nx] += w1;
    speeds_5_acc[ii + jj * params.nx] += w2;
    speeds_8_acc[ii + jj * params.nx] += w2;
    /* decrease 'west-side' densities */
    speeds_3_acc[ii + jj * params.nx] -= w1;
    speeds_6_acc[ii + jj * params.nx] -= w2;
    speeds_7_acc[ii + jj * params.nx] -= w2;
  }
}

__host__ __device__ inline void lbm_computation(
    accessor_t speeds_0_acc,
    accessor_t speeds_1_acc,
    accessor_t speeds_2_acc,
    accessor_t speeds_3_acc,
    accessor_t speeds_4_acc,
    accessor_t speeds_5_acc,
    accessor_t speeds_6_acc,
    accessor_t speeds_7_acc,
    accessor_t speeds_8_acc,
    accessor_t tmp_speeds_0_acc,   
    accessor_t tmp_speeds_1_acc,
    accessor_t tmp_speeds_2_acc,
    accessor_t tmp_speeds_3_acc,
    accessor_t tmp_speeds_4_acc,
    accessor_t tmp_speeds_5_acc,
    accessor_t tmp_speeds_6_acc,
    accessor_t tmp_speeds_7_acc,
    accessor_t tmp_speeds_8_acc,
    iread_accessor_t obstacles_acc,
    accessor_t tot_u_acc,
    accessor_t av_vels_acc, 
    cl::sycl::item<2> item,
    const t_param params)
{
  int ii = item[0];
  int jj = item[1];

  const float c_sq = 1.f / 3.f;  /* square of speed of sound */
  const float w0   = 4.f / 9.f;  /* weighting factor */
  const float w1   = 1.f / 9.f;  /* weighting factor */
  const float w2   = 1.f / 36.f; /* weighting factor */
  int   tot_cells  = 0;   /* no. of cells used in calculation */
  float tot_u      = 0.f; /* accumulated magnitudes of velocity for each cell */

  // PROPAGATION STEP:
  /* determine indices of axis-direction neighbours
  ** respecting periodic boundary conditions (wrap around) */
  const int y_n = (jj + 1) % params.ny;
  const int x_e = (ii + 1) % params.nx;
  const int y_s = (jj == 0) ? (jj + params.ny - 1) : (jj - 1);
  const int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);

  /* propagate densities from neighbouring cells, following
  ** appropriate directions of travel and writing into
  ** scratch space grid */
  const float s0 = speeds_0_acc[ii + jj*params.nx]; /* central cell, no movement */
  const float s1 = speeds_1_acc[x_w + jj*params.nx]; /* east */
  const float s2 = speeds_2_acc[ii + y_s*params.nx]; /* north */
  const float s3 = speeds_3_acc[x_e + jj*params.nx]; /* west */
  const float s4 = speeds_4_acc[ii + y_n*params.nx]; /* south */
  const float s5 = speeds_5_acc[x_w + y_s*params.nx]; /* north-east */
  const float s6 = speeds_6_acc[x_e + y_s*params.nx]; /* north-west */
  const float s7 = speeds_7_acc[x_e + y_n*params.nx]; /* south-west */
  const float s8 = speeds_8_acc[x_w + y_n*params.nx]; /* south-east */

  // COLLISION STEP:
  /* compute local density total */
  const float local_density = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8;

  /* compute x velocity component */
  const float u_x = (s1 + s5 + s8 - (s3 + s6 + s7)) / local_density;

  /* compute y velocity component */
  const float u_y = (s2 + s5 + s6 - (s4 + s7 + s8)) / local_density;

  /* velocity squared */
  const float u_sq = u_x * u_x + u_y * u_y;

  /* directional velocity components */
  float u[NSPEEDS];
  u[1] =   u_x;        /* east */
  u[2] =         u_y;  /* north */
  u[3] = - u_x;        /* west */
  u[4] =       - u_y;  /* south */
  u[5] =   u_x + u_y;  /* north-east */
  u[6] = - u_x + u_y;  /* north-west */
  u[7] = - u_x - u_y;  /* south-west */
  u[8] =   u_x - u_y;  /* south-east */

          /* equilibrium densities */
          float d_equ[NSPEEDS];
          /* zero velocity density: weight w0 */
          d_equ[0] = w0 * local_density
                     * (1.f - u_sq / (2.f * c_sq));
          /* axis speeds: weight w1 */
          d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                           + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                           - u_sq / (2.f * c_sq));
          d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                           + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                                           - u_sq / (2.f * c_sq));
          d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                           + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                                           - u_sq / (2.f * c_sq));
          d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                           + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                                           - u_sq / (2.f * c_sq));
          /* diagonal speeds: weight w2 */
          d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                           + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                                           - u_sq / (2.f * c_sq));
          d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                           + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                                           - u_sq / (2.f * c_sq));
          d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                           + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                                           - u_sq / (2.f * c_sq));
          d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                           + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                                           - u_sq / (2.f * c_sq));

          /* relaxation step */
          const float t0 = (obstacles_acc[jj*params.nx + ii] != 0) ? s0 : (s0 + params.omega * (d_equ[0] - s0));
          const float t1 = (obstacles_acc[jj*params.nx + ii] != 0) ? s3 : (s1 + params.omega * (d_equ[1] - s1));
          const float t2 = (obstacles_acc[jj*params.nx + ii] != 0) ? s4 : (s2 + params.omega * (d_equ[2] - s2));
          const float t3 = (obstacles_acc[jj*params.nx + ii] != 0) ? s1 : (s3 + params.omega * (d_equ[3] - s3));
          const float t4 = (obstacles_acc[jj*params.nx + ii] != 0) ? s2 : (s4 + params.omega * (d_equ[4] - s4));
          const float t5 = (obstacles_acc[jj*params.nx + ii] != 0) ? s7 : (s5 + params.omega * (d_equ[5] - s5));
          const float t6 = (obstacles_acc[jj*params.nx + ii] != 0) ? s8 : (s6 + params.omega * (d_equ[6] - s6));
          const float t7 = (obstacles_acc[jj*params.nx + ii] != 0) ? s5 : (s7 + params.omega * (d_equ[7] - s7));
          const float t8 = (obstacles_acc[jj*params.nx + ii] != 0) ? s6 : (s8 + params.omega * (d_equ[8] - s8));

          // AVERAGE VELOCITIES STEP:
          /* local density total */
          const float local_density_v = t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8;

          /* x-component of velocity */
          const float u_x_v = (t1 + t5 + t8 - (t3 + t6 + t7)) / local_density_v;
          /* compute y velocity component */
          const float u_y_v = (t2 + t5 + t6 - (t4 + t7 + t8)) / local_density_v;

          /* accumulate the norm of x- and y- velocity components */
          tot_u_acc[jj*params.nx + ii] = (obstacles_acc[jj*params.nx + ii] != 0) ? 0 : sqrtf((u_x_v * u_x_v) + (u_y_v * u_y_v));

          tmp_speeds_0_acc[ii + jj*params.nx] = t0;
          tmp_speeds_1_acc[ii + jj*params.nx] = t1;
          tmp_speeds_2_acc[ii + jj*params.nx] = t2;
          tmp_speeds_3_acc[ii + jj*params.nx] = t3;
          tmp_speeds_4_acc[ii + jj*params.nx] = t4;
          tmp_speeds_5_acc[ii + jj*params.nx] = t5;
          tmp_speeds_6_acc[ii + jj*params.nx] = t6;
          tmp_speeds_7_acc[ii + jj*params.nx] = t7;
          tmp_speeds_8_acc[ii + jj*params.nx] = t8;

}

int timestep(const t_param params, t_soa* cells, t_soa* tmp_cells, int* obstacles, float* av_vels, const int tt) {
  accelerate_flow(params, cells, obstacles);
  av_vels[tt] = prop_rebound_collision_avels(params, cells, tmp_cells, obstacles);
#ifdef DEBUG
  printf("==timestep: %d==\n", tt);
  printf("av velocity: %.12E\n", av_vels[tt]);
  printf("tot density: %.12E\n", total_density(params, tmp_cells));
#endif
  return EXIT_SUCCESS;
}

int accelerate_flow(const t_param params, t_soa* cells, int* obstacles) {
  /* compute weighting factors */
  const float w1 = params.density * params.accel / 9.f;
  const float w2 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */
  const int jj = params.ny - 2;

  for (int ii = 0; ii < params.nx; ii++) {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii + jj*params.nx]
        && (cells->speed_3[ii + jj*params.nx] - w1) > 0.f
        && (cells->speed_6[ii + jj*params.nx] - w2) > 0.f
        && (cells->speed_7[ii + jj*params.nx] - w2) > 0.f) {
      /* increase 'east-side' densities */
      cells->speed_1[ii + jj*params.nx] += w1;
      cells->speed_5[ii + jj*params.nx] += w2;
      cells->speed_8[ii + jj*params.nx] += w2;
      /* decrease 'west-side' densities */
      cells->speed_3[ii + jj*params.nx] -= w1;
      cells->speed_6[ii + jj*params.nx] -= w2;
      cells->speed_7[ii + jj*params.nx] -= w2;
    }
  }

  return EXIT_SUCCESS;
}

float prop_rebound_collision_avels(const t_param params, t_soa* cells, t_soa* tmp_cells, int* obstacles) {
  const float c_sq = 1.f / 3.f;  /* square of speed of sound */
  const float w0   = 4.f / 9.f;  /* weighting factor */
  const float w1   = 1.f / 9.f;  /* weighting factor */
  const float w2   = 1.f / 36.f; /* weighting factor */
  int   tot_cells  = 0;   /* no. of cells used in calculation */
  float tot_u      = 0.f; /* accumulated magnitudes of velocity for each cell */

  /* loop over the cells in the grid */
  for (int jj = 0; jj < params.ny; jj++) {
    for (int ii = 0; ii < params.nx; ii++) {
      // PROPAGATION STEP:
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      const int y_n = (jj + 1) % params.ny;
      const int x_e = (ii + 1) % params.nx;
      const int y_s = (jj == 0) ? (jj + params.ny - 1) : (jj - 1);
      const int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);

      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      const float s0 = cells->speed_0[ii + jj*params.nx]; /* central cell, no movement */
      const float s1 = cells->speed_1[x_w + jj*params.nx]; /* east */
      const float s2 = cells->speed_2[ii + y_s*params.nx]; /* north */
      const float s3 = cells->speed_3[x_e + jj*params.nx]; /* west */
      const float s4 = cells->speed_4[ii + y_n*params.nx]; /* south */
      const float s5 = cells->speed_5[x_w + y_s*params.nx]; /* north-east */
      const float s6 = cells->speed_6[x_e + y_s*params.nx]; /* north-west */
      const float s7 = cells->speed_7[x_e + y_n*params.nx]; /* south-west */
      const float s8 = cells->speed_8[x_w + y_n*params.nx]; /* south-east */

      // COLLISION STEP:
      /* compute local density total */
      const float local_density = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8;

      /* compute x velocity component */
      const float u_x = (s1 + s5 + s8 - (s3 + s6 + s7)) / local_density;

      /* compute y velocity component */
      const float u_y = (s2 + s5 + s6 - (s4 + s7 + s8)) / local_density;

      /* velocity squared */
      const float u_sq = u_x * u_x + u_y * u_y;

      /* directional velocity components */
      float u[NSPEEDS];
      u[1] =   u_x;        /* east */
      u[2] =         u_y;  /* north */
      u[3] = - u_x;        /* west */
      u[4] =       - u_y;  /* south */
      u[5] =   u_x + u_y;  /* north-east */
      u[6] = - u_x + u_y;  /* north-west */
      u[7] = - u_x - u_y;  /* south-west */
      u[8] =   u_x - u_y;  /* south-east */

        /* equilibrium densities */
        float d_equ[NSPEEDS];
        /* zero velocity density: weight w0 */
        d_equ[0] = w0 * local_density
                   * (1.f - u_sq / (2.f * c_sq));
        /* axis speeds: weight w1 */
        d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                         + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                         + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                         + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                         + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        /* diagonal speeds: weight w2 */
        d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                         + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                         + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                         + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                         + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));

        /* relaxation step */
        const float t0 = (obstacles[jj*params.nx + ii] != 0) ? s0 : (s0 + params.omega * (d_equ[0] - s0));
        const float t1 = (obstacles[jj*params.nx + ii] != 0) ? s3 : (s1 + params.omega * (d_equ[1] - s1));
        const float t2 = (obstacles[jj*params.nx + ii] != 0) ? s4 : (s2 + params.omega * (d_equ[2] - s2));
        const float t3 = (obstacles[jj*params.nx + ii] != 0) ? s1 : (s3 + params.omega * (d_equ[3] - s3));
        const float t4 = (obstacles[jj*params.nx + ii] != 0) ? s2 : (s4 + params.omega * (d_equ[4] - s4));
        const float t5 = (obstacles[jj*params.nx + ii] != 0) ? s7 : (s5 + params.omega * (d_equ[5] - s5));
        const float t6 = (obstacles[jj*params.nx + ii] != 0) ? s8 : (s6 + params.omega * (d_equ[6] - s6));
        const float t7 = (obstacles[jj*params.nx + ii] != 0) ? s5 : (s7 + params.omega * (d_equ[7] - s7));
        const float t8 = (obstacles[jj*params.nx + ii] != 0) ? s6 : (s8 + params.omega * (d_equ[8] - s8));

        // AVERAGE VELOCITIES STEP:
        /* local density total */
        const float local_density_v = t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8;

        /* x-component of velocity */
        const float u_x_v = (t1 + t5 + t8 - (t3 + t6 + t7)) / local_density_v;
        /* compute y velocity component */
        const float u_y_v = (t2 + t5 + t6 - (t4 + t7 + t8)) / local_density_v;

        /* accumulate the norm of x- and y- velocity components */
        tot_u += (obstacles[jj*params.nx + ii] != 0) ? 0 : sqrtf((u_x_v * u_x_v) + (u_y_v * u_y_v));
        /* increase counter of inspected cells */
        tot_cells += (obstacles[jj*params.nx + ii] != 0) ? 0 : 1;

        tmp_cells->speed_0[ii + jj*params.nx] = t0;
        tmp_cells->speed_1[ii + jj*params.nx] = t1;
        tmp_cells->speed_2[ii + jj*params.nx] = t2;
        tmp_cells->speed_3[ii + jj*params.nx] = t3;
        tmp_cells->speed_4[ii + jj*params.nx] = t4;
        tmp_cells->speed_5[ii + jj*params.nx] = t5;
        tmp_cells->speed_6[ii + jj*params.nx] = t6;
        tmp_cells->speed_7[ii + jj*params.nx] = t7;
        tmp_cells->speed_8[ii + jj*params.nx] = t8;
    }
  }

  return tot_u / (float)tot_cells;
}

float av_velocity(const t_param params, t_soa* cells, int* obstacles) {
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

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_soa** cells_ptr, t_soa** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr) {
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

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

  (*cells_ptr)->speed_0 = (float*) malloc(sizeof(float) * (params->ny * params->nx));
  (*cells_ptr)->speed_1 = (float*) malloc(sizeof(float) * (params->ny * params->nx));
  (*cells_ptr)->speed_2 = (float*) malloc(sizeof(float) * (params->ny * params->nx));
  (*cells_ptr)->speed_3 = (float*) malloc(sizeof(float) * (params->ny * params->nx));
  (*cells_ptr)->speed_4 = (float*) malloc(sizeof(float) * (params->ny * params->nx));
  (*cells_ptr)->speed_5 = (float*) malloc(sizeof(float) * (params->ny * params->nx));
  (*cells_ptr)->speed_6 = (float*) malloc(sizeof(float) * (params->ny * params->nx));
  (*cells_ptr)->speed_7 = (float*) malloc(sizeof(float) * (params->ny * params->nx));
  (*cells_ptr)->speed_8 = (float*) malloc(sizeof(float) * (params->ny * params->nx));

  /* 'helper' grid, used as scratch space */
  *tmp_cells_ptr = (t_soa*) malloc(sizeof(t_soa));

  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  (*tmp_cells_ptr)->speed_0 = (float*) malloc(sizeof(float) * (params->ny * params->nx));
  (*tmp_cells_ptr)->speed_1 = (float*) malloc(sizeof(float) * (params->ny * params->nx));
  (*tmp_cells_ptr)->speed_2 = (float*) malloc(sizeof(float) * (params->ny * params->nx));
  (*tmp_cells_ptr)->speed_3 = (float*) malloc(sizeof(float) * (params->ny * params->nx));
  (*tmp_cells_ptr)->speed_4 = (float*) malloc(sizeof(float) * (params->ny * params->nx));
  (*tmp_cells_ptr)->speed_5 = (float*) malloc(sizeof(float) * (params->ny * params->nx));
  (*tmp_cells_ptr)->speed_6 = (float*) malloc(sizeof(float) * (params->ny * params->nx));
  (*tmp_cells_ptr)->speed_7 = (float*) malloc(sizeof(float) * (params->ny * params->nx));
  (*tmp_cells_ptr)->speed_8 = (float*) malloc(sizeof(float) * (params->ny * params->nx));

  /* the map of obstacles */
  *obstacles_ptr = (int*) malloc(sizeof(int) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density       / 9.f;
  float w2 = params->density       / 36.f;

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

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_soa** cells_ptr, t_soa** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr) {
  // free up allocated memory
  // free(cells_ptr);
  // *cells_ptr = NULL;
  //
  // free(*tmp_cells_ptr);
  // *tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, t_soa* cells, int* obstacles) {
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
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
