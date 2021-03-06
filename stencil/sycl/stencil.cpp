#include <iostream>
#include <CL/sycl.hpp>
#include <sys/time.h>

class stencil_op1;
class stencil_op2;

// Define output file name
#define OUTPUT_FILE "stencil.pgm"

void init_image(const int nx, const int ny, float* image, float* tmp_image);
void stencil(const int nx, const int ny, float* image, float* tmp_image);
void output_image(const char* file_name, const int nx, const int ny, float* image);
double wtime(void);

int main(int argc, char* argv[]) {
  
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
  float* image     = (float*) malloc(sizeof(float)*(nx+2)*(ny+2));
  float* tmp_image = (float*) malloc(sizeof(float)*(nx+2)*(ny+2));
  
  // Set the input image
  init_image(nx+2, ny+2, image, tmp_image);

  cl::sycl::default_selector device_selector;

  cl::sycl::queue queue(device_selector);
  std::cout << "Running on "
            << queue.get_device().get_info<cl::sycl::info::device::name>()
            << "\n";

  double tic = wtime(); 
  {
    cl::sycl::buffer<float, 1> img_sycl(image, cl::sycl::range<1>((nx + 2) * (ny + 2)));
    cl::sycl::buffer<float, 1> tmp_img_sycl(tmp_image, cl::sycl::range<1>((nx + 2) * (ny + 2)));

    for(int t = 0; t < niters; ++t) {
      queue.submit([&] (cl::sycl::handler& cgh) {
        auto img_acc = img_sycl.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto tmp_img_acc = tmp_img_sycl.get_access<cl::sycl::access::mode::read_write>(cgh);

        cgh.parallel_for<class stencil_op1>(cl::sycl::range<2>(nx, ny), [=](cl::sycl::item<2> item) {
          int i = item[0] + 1;
          int j = item[1] + 1;
          int sz = nx + 2;
          
          tmp_img_acc[ j + i * sz ] = img_acc[ j + i * sz ] * 0.6f +
                                    ( img_acc[ j - 1 + i * sz ]    +
                                      img_acc[ j + 1 + i * sz ]    +
                                      img_acc[ j + (i - 1) * sz ]  +
                                      img_acc[ j + (i + 1) * sz ] ) * 0.1f;
        });
        cgh.parallel_for<class stencil_op2>(cl::sycl::range<2>(nx, ny), [=](cl::sycl::item<2> item) {
          int i = item[0] + 1;
          int j = item[1] + 1;
          int sz = nx + 2;
          
          img_acc[ j + i * sz ] = tmp_img_acc[ j + i * sz ] * 0.6f +
                                ( tmp_img_acc[ j - 1 + i * sz ]    +
                                  tmp_img_acc[ j + 1 + i * sz ]    +
                                  tmp_img_acc[ j + (i - 1) * sz ]  +
                                  tmp_img_acc[ j + (i + 1) * sz ] ) * 0.1f;
        });
      });
    }
  } 
  double toc = wtime();

  // Output
  printf("----------------------------------------\n");
  printf(" runtime:          %lf s\n", toc-tic);
  printf(" memory bandwidth: %lf GB/s\n", (4 * 6 * (nx / 1024) * (ny / 1024) * 2 * niters) / ((toc - tic) * 1024) );
  printf("----------------------------------------\n");
  output_image(OUTPUT_FILE, nx+2, ny+2, image);
  free(image);
  
   
  return 0;
}

// Create the input image
void init_image(const int nx, const int ny, float* image, float* tmp_image) {
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

// Routine to output the image in Netpbm grayscale binary image format
void output_image(const char* file_name, const int nx, const int ny, float* image) {

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
