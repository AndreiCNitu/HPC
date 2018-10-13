/*
 * v0 : Original version.
 * v1 : Changed from column major to row major order in stencil function
 * v2 : Replaced arithmetic divisions( 3.0/5.0, 0.5/5.0 ) with
 * double values( 0.6, 0.1 )
 * v2.1 : Use compiler flags( gcc/icc O1, O2 ,O3 ,Ofast ... )
 * v3 : Use tmp_row instead of tmp_image( save memory )
 * - change init_image to only initialise the main image to checkerboard
 * - change the timed forloop to go to 2 * niters and use tmp_row
 * - change stencil function to use tmp_row, also save current cell and reset it
 *   in tmp_row to use for the next cell to the right
 * v4 :
*/
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Define output file name
#define OUTPUT_FILE "stencil.pgm"

void stencil(const int nx, const int ny, double * restrict image, double * restrict tmp_row);
void init_image(const int nx, const int ny, double * image, double * tmp_row);
void output_image(const char * file_name, const int nx, const int ny, double *image);
double wtime(void);

int main(int argc, char *argv[]) {

  // Check usage
  if (argc != 4) {
    fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  // Initiliase problem dimensions from command line arguments
  int nx = atoi(argv[1]);
  int ny = atoi(argv[2]);
  int niters = atoi(argv[3]);

  // Allocate the image
  double *image = malloc( sizeof(double) * (nx+2) * (ny+2) );
  double *tmp_row = malloc( sizeof(double) * (nx+2) * (ny+2) );

  // Set the input image
  init_image(nx, ny, image, tmp_row);

  // Call the stencil kernel
  double tic = wtime();
  for (int t = 0; t < 2 * niters; ++t) {
    stencil(nx, ny, image, tmp_row);
  }
  double toc = wtime();


  // Output
  printf("------------------------------------\n");
  printf(" runtime: %lf s\n", toc-tic);
  printf("------------------------------------\n");

  output_image(OUTPUT_FILE, nx, ny, image);
  free(image);
}

void stencil(const int nx, const int ny, double * restrict image, double * restrict tmp_row) {
  double current_cell;
  for( int i = 1; i < nx+1; i++ ) {
    for( int j = 1; j < ny+1; j++ ) {
      current_cell = image[ j + i * ny ];
      image[ j + i * ny ] *= 0.6;
      if (i > 0)    image[ j + i * ny ] += tmp_row[ j ] * 0.1;
      if (i < nx-1) image[ j + i * ny ] += image[ j + (i + 1) * ny ] * 0.1;
      if (j > 0)    image[ j + i * ny ] += tmp_row[ j - 1  ] * 0.1;
      if (j < ny-1) image[ j + i * ny ] += image[ j + 1 + i * ny ] * 0.1;
      tmp_row[ j ] = current_cell;
    }
  }
}

// Create the input image
void init_image(const int nx, const int ny, double * image, double * tmp_row) {
  // Zero everything
  for (int j = 0; j < ny+2; ++j) {
    for (int i = 0; i < nx+2; ++i) {
      image[j+i*ny] = 0.0;
    }
  }

  // Initialize temporary row
  for (int i = 0; i < ny+2; ++i)
    tmp_row[i] = 0.0;

  // Checkerboard
  for (int j = 0; j < 8; ++j) {
    for (int i = 0; i < 8; ++i) {
      for (int jj = j*ny/8; jj < (j+1)*ny/8; ++jj) {
        for (int ii = i*nx/8; ii < (i+1)*nx/8; ++ii) {
          if ((i+j)%2)
            image[ (jj + 1) + (ii + 1) * ny ] = 100.0;
        }
      }
    }
  }
}

// Routine to output the image in Netpbm grayscale binary image format
void output_image(const char * file_name, const int nx, const int ny, double *image) {

  // Open output file
  FILE *fp = fopen(file_name, "w");
  if (!fp) {
    fprintf(stderr, "Error: Could not open %s\n", OUTPUT_FILE);
    exit(EXIT_FAILURE);
  }

  // Ouptut image header
  fprintf(fp, "P5 %d %d 255\n", nx, ny);

  // Calculate maximum value of image
  // This is used to rescale the values
  // to a range of 0-255 for output
  double maximum = 0.0;
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      if (image[j+i*ny] > maximum)
        maximum = image[j+i*ny];
    }
  }

  // Output image, converting to numbers 0-255
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
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
