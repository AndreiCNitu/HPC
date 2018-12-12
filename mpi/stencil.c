
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>

// Define output file name
#define OUTPUT_FILE "stencil.pgm"

void stencil( const int nx, const int ny, float * restrict image, float * restrict tmp_image );
void init_images( const int nx, const int ny, float * image, float * tmp_image );
void init_proc_images( const int nx, const int ny, float *image, float *proc_image, float *tmp_proc_image, int p_start, int p_end, int rank );
void comm_neighbours( int nx, int ny, float *image, int rank, int size );
void output_image( const char * file_name, const int nx, const int ny, float *image );
double wtime(void);

int main(int argc, char *argv[]) {

  // Check usage
  if (argc != 4) {
    fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  int rank; // Rank of process among its cohort 
  int size; // Size of cohort (Number of processes)
  int flag; // For checking whether MPI_Init() has been called
  int strlen;
  enum bool {FALSE, TRUE};
  char hostname[MPI_MAX_PROCESSOR_NAME];

  // Initialise MPI environment
  MPI_Init( &argc, &argv );

  // Check whether the initialisation was successful
  MPI_Initialized( &flag );
  if( flag != TRUE ) {
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  // Determine hostname
  MPI_Get_processor_name(hostname, &strlen);

  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank); 

  // Initiliase problem dimensions from command line arguments
  int nx = atoi(argv[1]); // ROWS
  int ny = atoi(argv[2]); // COLS
  int niters = atoi(argv[3]);

  // Allocate the full image
  float *image     = _mm_malloc(sizeof(float) * (nx + 2) * (ny + 2), 64);
  float *tmp_image = _mm_malloc(sizeof(float) * (nx + 2) * (ny + 2), 64);

  // Set the input image
  init_images(nx + 2, ny + 2, image, tmp_image);
    
  // Initialise process matrix = ny / size + 2 rows
  int p_height = (nx / size) + 2;        // Number of rows process holds 
  int p_start  = rank * (p_height - 2);  // First row (inclusive)
  int p_end    = p_start + p_height - 1; // Last  row (inclusive)

  float *proc_image     = _mm_malloc(sizeof(float) * (nx + 2) * p_height, 64);
  float *tmp_proc_image = _mm_malloc(sizeof(float) * (nx + 2) * p_height, 64);
  
  init_proc_images(nx+2, ny+2, image, proc_image, tmp_proc_image, p_start, p_end, rank);

  // Call the stencil kernel
  double tic = wtime();
  //for (int t = 0; t < niters; ++t) {   
    //stencil(p_height, ny+2, proc_image, tmp_proc_image);
    //comm_neighbours(p_height, ny+2, tmp_proc_image, rank, size);
   
    //stencil(p_height, ny+2, tmp_proc_image, proc_image);
    //comm_neighbours(p_height, ny+2, proc_image, rank, size);
  //}
  double toc = wtime();

  // Compute memory bandwidth usage
  printf("----------------------------------------\n");
  printf(" runtime:          %lf s\n", toc-tic);
  printf(" memory bandwidth: %lf GB/s\n", (4 * 6 * (nx / 1024) * (ny / 1024) * 2 * niters) / ((toc - tic) * 1024) );
  printf("----------------------------------------\n");

  float *out_image = _mm_malloc(sizeof(float) * (nx + 2) * (ny + 2), 64); 
  if(rank == 0) {
    for(int i = 0; i < nx + 2; i++) {
      for(int j = 0; j < ny + 2; j++) {
        out_image[j + i * (ny + 2)] = 0.0; 
      }
    }

    for (int i = 1; i < nx + 1; ++i) {
      for (int j = 0; j < ny + 2; ++j) {
        image[j+i*(ny+2)] = proc_image[j+i*(ny+2)];
      }
    }

    for(int pid = 1; pid < size; pid++) {
      MPI_Recv((float*) out_image + (ny + 2) * (pid * (p_height-2) + 1), (ny+2) * (p_height-2), MPI_FLOAT, pid, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      printf("Received from P%d\n", pid);
    }
  } else {
    MPI_Ssend((float*) proc_image + ny + 2, (ny+2) * (p_height-2), MPI_FLOAT, 0, 42, MPI_COMM_WORLD);
  }
    if(rank == 0) {
      output_image(OUTPUT_FILE, nx+2, ny+2, out_image);
    }
  _mm_free(image);

  MPI_Finalize();

  return EXIT_SUCCESS;
}

void comm_neighbours( int nx, int ny, float *image, int rank, int size ) {
  // even -> send, recv
  // odd  -> recv, send
  
  if       (rank == 0) {
    // Send row nx-2 DOWN
    MPI_Send((float*) image + ny * (nx-2), ny, MPI_FLOAT, rank+1, 42, MPI_COMM_WORLD);
    // Recv row nx-1 DOWN
    MPI_Recv((float*) image + ny * (nx-1), ny, MPI_FLOAT, rank+1, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  } else if(rank == size-1) {
    // Send row 1 UP
    MPI_Send((float*) image + ny,          ny, MPI_FLOAT, rank-1, 42, MPI_COMM_WORLD);
    // Recv row 0 UP
    MPI_Recv((float*) image,               ny, MPI_FLOAT, rank-1, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  } else { // middle
    if(rank % 2 == 0) {
      // Send row 1 UP, row nx-2 DOWN
      MPI_Send((float*) image + ny,          ny, MPI_FLOAT, rank-1, 42, MPI_COMM_WORLD);
      MPI_Send((float*) image + ny * (nx-2), ny, MPI_FLOAT, rank+1, 42, MPI_COMM_WORLD);
      // Recv row 0 UP, row nx-1 DOWN
      MPI_Recv((float*) image,               ny, MPI_FLOAT, rank-1, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv((float*) image + ny * (nx-1), ny, MPI_FLOAT, rank+1, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
      // Recv row 0 UP, row nx-1 DOWN
      MPI_Recv((float*) image,               ny, MPI_FLOAT, rank-1, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv((float*) image + ny * (nx-1), ny, MPI_FLOAT, rank+1, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      // Send row 1 UP, row nx-2 DOWN
      MPI_Send((float*) image + ny,          ny, MPI_FLOAT, rank-1, 42, MPI_COMM_WORLD);
      MPI_Send((float*) image + ny * (nx-2), ny, MPI_FLOAT, rank+1, 42, MPI_COMM_WORLD);
    }
  }
}

void stencil(const int nx, const int ny, float * restrict image, float * restrict tmp_image) {
  for( int i = 1; i < nx-1; i++ ) {
    __assume_aligned(tmp_image, 64);
    __assume_aligned(    image, 64);
    __assume( ((i - 1) * ny) % 16 == 0 );
    __assume( ((i + 1) * ny) % 16 == 0 );
    __assume( (-1 + i * ny)  % 16 == 0 );
    __assume( ( 1 + i * ny)  % 16 == 0 );
    #pragma simd
    
    #pragma unroll (4)
    for( int j = 1; j < ny-1; j++ ) {
      tmp_image[ j + i * ny ]  = image[ j + i * ny ] * 0.6f +
                               ( image[ j - 1 + i * ny ]    +
                                 image[ j + 1 + i * ny ]    +
                                 image[ j + (i - 1) * ny ]  +
                                 image[ j + (i + 1) * ny ] ) * 0.1f;
    }
  }
}

// Create the input image
void init_images(const int nx, const int ny, float * image, float * tmp_image) {
  // Zero everything
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      image[j+i*ny] = 0.0;
    }
  }

  // Initialize temporary image with 0s
  for (int i = 0; i < ny; ++i)
    for (int j = 0; j < nx; ++j)
      tmp_image[j+i*ny] = 0.0;

  // Checkerboard --- margins = 0
  for (int j = 0; j < 8; ++j) {
    for (int i = 0; i < 8; ++i) {
      for (int jj = j*(ny-2)/8; jj < (j+1)*(ny-2)/8; ++jj) {
        for (int ii = i*(nx-2)/8; ii < (i+1)*(nx-2)/8; ++ii) {
          if ((i+j)%2)
            image[(jj+1)+(ii+1)*ny] = 100.0;
        }
      }
    }
  }

}

void init_proc_images( const int nx, const int ny, float *image, float *proc_image, float *tmp_proc_image, int p_start, int p_end, int rank ) {
  // Initialise temporary image with 0s
  for (int i = 0; i <= p_end - p_start; ++i) {
    for (int j = 0; j < ny; ++j) {
      tmp_proc_image[j+i*ny] = 0.0;
    }
  }

  // Copy image section
  for (int i = p_start; i <= p_end; ++i) {
    for (int j = 0; j < ny; ++j) {
      proc_image[j+(i-p_start)*ny] = image[j+i*ny];   
    }
  }

}

// Routine to output the image in Netpbm grayscale binary image format
void output_image(const char * file_name, const int nx, const int ny, float *image) {

  // Open output file
  FILE *fp = fopen(file_name, "w");
  if (!fp) {
    fprintf(stderr, "Error: Could not open %s\n", OUTPUT_FILE);
    exit(EXIT_FAILURE);
  }

  // Ouptut image header
  fprintf(fp, "P5 %d %d 255\n", nx-2, ny-2);

  // Calculate maximum value of image
  // This is used to rescale the values
  // to a range of 0-255 for output
  float maximum = 0.0;
  for (int j = 1; j < ny-1; ++j) {
    for (int i = 1; i < nx-1; ++i) {
      if (image[j+i*ny] > maximum)
        maximum = image[j+i*ny];
    }
  }

  // Output image, converting to numbers 0-255
  for (int j = 1; j < ny-1; ++j) {
    for (int i = 1; i < nx-1; ++i) {
      fputc((char)(255.0*image[j+i*ny]/maximum), fp);
    }
  }

  // Close the file
  fclose(fp);

}

// Get the current time in seconds since the Epoch
double wtime(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec*1e-6;
}
