#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS 9

#define c_sq 0.3333333333333333f  /* square of speed of sound */
#define W0   0.4444444444444444f  /* weighting factor */
#define W1   0.1111111111111111f  /* weighting factor */
#define W2   0.0277777777777777f /* weighting factor */


kernel void accelerate_flow(global float* restrict cells_speed_0,
                            global float* restrict cells_speed_1,
                            global float* restrict cells_speed_2,
                            global float* restrict cells_speed_3,
                            global float* restrict cells_speed_4,
                            global float* restrict cells_speed_5,
                            global float* restrict cells_speed_6,
                            global float* restrict cells_speed_7,
                            global float* restrict cells_speed_8,
                            global int*   restrict obstacles,
                            const int nx, const int ny,
                            const float density, const float accel) {

  /* compute weighting factors */
  const float w1 = density * accel / 9.0;
  const float w2 = density * accel / 36.0;

  /* modify the 2nd row of the grid */
  const int jj = ny - 2;

  /* get column index */
  const int ii = get_global_id(0);

  /* if the cell is not occupied and
  ** we don't send a negative density */
  /* if the cell is not occupied and
  ** we don't send a negative density */
  if (!obstacles[ii + jj*nx]
      && (cells_speed_3[ii + jj*nx] - w1) > 0.f
      && (cells_speed_6[ii + jj*nx] - w2) > 0.f
      && (cells_speed_7[ii + jj*nx] - w2) > 0.f) {
    /* increase 'east-side' densities */
    cells_speed_1[ii + jj*nx] += w1;
    cells_speed_5[ii + jj*nx] += w2;
    cells_speed_8[ii + jj*nx] += w2;
    /* decrease 'west-side' densities */
    cells_speed_3[ii + jj*nx] -= w1;
    cells_speed_6[ii + jj*nx] -= w2;
    cells_speed_7[ii + jj*nx] -= w2;
  }

}

kernel void reduce_vels( global float* partial_tot_u,
                         global int*   partial_tot_cells,
                         global float* av_vels,
                         const int num_wrks,
                         const int tt) {

    float total_u = 0.0f;
    int total_cells = 0;
    for (unsigned long i = 0; i < num_wrks; i++) {
      total_u += partial_tot_u[i];
      total_cells += partial_tot_cells[i];
    }
    av_vels[tt] = total_u / (float)total_cells;
}

kernel void prop_rebound_collision_avels(global float* restrict cells_speed_0,
                                         global float* restrict cells_speed_1,
                                         global float* restrict cells_speed_2,
                                         global float* restrict cells_speed_3,
                                         global float* restrict cells_speed_4,
                                         global float* restrict cells_speed_5,
                                         global float* restrict cells_speed_6,
                                         global float* restrict cells_speed_7,
                                         global float* restrict cells_speed_8,
                                         global float* restrict tmp_cells_speed_0,
                                         global float* restrict tmp_cells_speed_1,
                                         global float* restrict tmp_cells_speed_2,
                                         global float* restrict tmp_cells_speed_3,
                                         global float* restrict tmp_cells_speed_4,
                                         global float* restrict tmp_cells_speed_5,
                                         global float* restrict tmp_cells_speed_6,
                                         global float* restrict tmp_cells_speed_7,
                                         global float* restrict tmp_cells_speed_8,
                                         global int*   restrict obstacles,
                                         const int nx, const int ny, const int tt, const float omega,
                                         global float* restrict partial_tot_u,
                                         global int*   restrict partial_tot_cells,
                                         local  float* restrict local_tot_u,
                                         local  int*   restrict local_tot_cells) {


  /* get column and row indices */
  const int ii = get_global_id(0);
  const int jj = get_global_id(1);
  const int l_ii = get_local_id(0);
  const int l_jj = get_local_id(1);
  const int g_ii = get_group_id(0);
  const int g_jj = get_group_id(1);

  const int local_nx = get_local_size(0);
  const int local_ny = get_local_size(1);

  // PROPAGATION STEP:
  /* determine indices of axis-direction neighbours
  ** respecting periodic boundary conditions (wrap around) */
  const int y_n = (jj + 1) % ny;
  const int x_e = (ii + 1) % nx;
  const int y_s = (jj == 0) ? (jj + ny - 1) : (jj - 1);
  const int x_w = (ii == 0) ? (ii + nx - 1) : (ii - 1);

  /* propagate densities from neighbouring cells, following
  ** appropriate directions of travel and writing into
  ** scratch space grid */
  const float s0 = cells_speed_0[ii + jj*nx]; /* central cell, no movement */
  const float s1 = cells_speed_1[x_w + jj*nx]; /* east */
  const float s2 = cells_speed_2[ii + y_s*nx]; /* north */
  const float s3 = cells_speed_3[x_e + jj*nx]; /* west */
  const float s4 = cells_speed_4[ii + y_n*nx]; /* south */
  const float s5 = cells_speed_5[x_w + y_s*nx]; /* north-east */
  const float s6 = cells_speed_6[x_e + y_s*nx]; /* north-west */
  const float s7 = cells_speed_7[x_e + y_n*nx]; /* south-west */
  const float s8 = cells_speed_8[x_w + y_n*nx]; /* south-east */
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
  d_equ[0] = W0 * local_density
              * (1.f - u_sq / (2.f * c_sq));
  /* axis speeds: weight w1 */
  d_equ[1] = W1 * local_density * (1.f + u[1] / c_sq
                                    + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                    - u_sq / (2.f * c_sq));
  d_equ[2] = W1 * local_density * (1.f + u[2] / c_sq
                                    + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                                    - u_sq / (2.f * c_sq));
  d_equ[3] = W1 * local_density * (1.f + u[3] / c_sq
                                    + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                                    - u_sq / (2.f * c_sq));
  d_equ[4] = W1 * local_density * (1.f + u[4] / c_sq
                                    + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                                    - u_sq / (2.f * c_sq));
  /* diagonal speeds: weight w2 */
  d_equ[5] = W2 * local_density * (1.f + u[5] / c_sq
                                    + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                                    - u_sq / (2.f * c_sq));
  d_equ[6] = W2 * local_density * (1.f + u[6] / c_sq
                                    + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                                    - u_sq / (2.f * c_sq));
  d_equ[7] = W2 * local_density * (1.f + u[7] / c_sq
                                    + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                                    - u_sq / (2.f * c_sq));
  d_equ[8] = W2 * local_density * (1.f + u[8] / c_sq
                                    + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                                    - u_sq / (2.f * c_sq));

  /* relaxation step */
  const float t0 = (obstacles[jj*nx + ii] != 0) ? s0 : (s0 + omega * (d_equ[0] - s0));
  const float t1 = (obstacles[jj*nx + ii] != 0) ? s3 : (s1 + omega * (d_equ[1] - s1));
  const float t2 = (obstacles[jj*nx + ii] != 0) ? s4 : (s2 + omega * (d_equ[2] - s2));
  const float t3 = (obstacles[jj*nx + ii] != 0) ? s1 : (s3 + omega * (d_equ[3] - s3));
  const float t4 = (obstacles[jj*nx + ii] != 0) ? s2 : (s4 + omega * (d_equ[4] - s4));
  const float t5 = (obstacles[jj*nx + ii] != 0) ? s7 : (s5 + omega * (d_equ[5] - s5));
  const float t6 = (obstacles[jj*nx + ii] != 0) ? s8 : (s6 + omega * (d_equ[6] - s6));
  const float t7 = (obstacles[jj*nx + ii] != 0) ? s5 : (s7 + omega * (d_equ[7] - s7));
  const float t8 = (obstacles[jj*nx + ii] != 0) ? s6 : (s8 + omega * (d_equ[8] - s8));

  // AVERAGE VELOCITIES STEP:
  /* local density total */
  const float local_density_v = t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8;

  /* x-component of velocity */
  const float u_x_v = (t1 + t5 + t8 - (t3 + t6 + t7)) / local_density_v;
  /* compute y velocity component */
  const float u_y_v = (t2 + t5 + t6 - (t4 + t7 + t8)) / local_density_v;

  /* accumulate the norm of x- and y- velocity components */
  local_tot_u[l_ii + l_jj * local_nx] = (obstacles[jj*nx + ii] != 0) ? 0 : sqrt((u_x_v * u_x_v) + (u_y_v * u_y_v));
  /* increase counter of inspected cells */
  local_tot_cells[l_ii + l_jj * local_nx] = (obstacles[jj*nx + ii] != 0) ? 0 : 1;

  barrier(CLK_LOCAL_MEM_FENCE);

  tmp_cells_speed_0[ii + jj*nx] = t0;
  tmp_cells_speed_1[ii + jj*nx] = t1;
  tmp_cells_speed_2[ii + jj*nx] = t2;
  tmp_cells_speed_3[ii + jj*nx] = t3;
  tmp_cells_speed_4[ii + jj*nx] = t4;
  tmp_cells_speed_5[ii + jj*nx] = t5;
  tmp_cells_speed_6[ii + jj*nx] = t6;
  tmp_cells_speed_7[ii + jj*nx] = t7;
  tmp_cells_speed_8[ii + jj*nx] = t8;

  const int item_id = l_ii + local_nx * l_jj;
  for (int offset = local_nx * local_ny / 2; offset > 0; offset /= 2) { //??!! replace / with >>
    if (item_id < offset) {
      local_tot_u[item_id]     += local_tot_u[item_id + offset];
      local_tot_cells[item_id] += local_tot_cells[item_id + offset];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (item_id == 0) {
    const int base = tt * get_num_groups(0) * get_num_groups(1);
    partial_tot_u[base + g_ii + g_jj * get_num_groups(0)] = local_tot_u[0];
    partial_tot_cells[base + g_ii + g_jj * get_num_groups(0)] = local_tot_cells[0];
  }
}
