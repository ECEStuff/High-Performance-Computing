/**
 *  \file mandelbrot_cyclic.cc
 *
 *  \brief Implement your parallel mandelbrot set in this file.
 */

#include <iostream>
#include <cstdlib>
#include "render.hh"
#include <mpi.h>

int mandelbrot(double x, double y) {
  int maxit = 511;
  double cx = x;
  double cy = y;
  double newx, newy;

  int it = 0;
  for (it = 0; it < maxit && (x*x + y*y) < 4; ++it) {
    newx = x*x - y*y + cx;
    newy = 2*x*y + cy;
    x = newx;
    y = newy;
  }
  return it;
}

// start computation
void start(int *temp, double minX, double maxX, double minY, double maxY, double dx, double dy, int height, int width, int job_size, int rank, int size) {
     double x, y;

     // cyclic computation: e.g. with 4 processes, allocate as 0-4-8-12 | 1-5-9-13 | 2-6-10-14 | 3-7-11-15
     for (int i = 0, cy = rank; i < job_size && cy < height; i++, cy += size) { // job_size = height / size
	y = cy * dy + minY;
	x = minX;
	for (int j = 0; j < width; ++j) {
	  temp[i * width + j] = mandelbrot(x, y); // was temp[i * width + j]
          x += dx;
        }
     }
}

// copy the contiguous results to restore the original ordering, placed into the result matrix
void fixMatrix(int *results, int *permresults, int size, int height, int width, int job_height) {
     int cycNum = 0;
     int count = 0;

     for (int i = 0; i < job_height * size; i++) {
	for (int j = 0; j < width; j++) {
	  permresults[i * width + j] = results[(cycNum + count * job_height) * width + j];
	}
        count++;
	if (count >= size) {
	  cycNum++;
	  count %= size;
	}
     }
}

// compute remaining rows on process 0
void doRemainingRows(int *results, int data_size, int size, int height, int width, int job_height, double minX, double minY, double dx, double dy) {
     double x, y;
     y = job_height * size * dy;

     for (int i = job_height * size; i < height; i++) {
	x = minX;
	for (int j = 0; j < width; j++) {
	  results[i * width + j] = mandelbrot(x,y);
	  x += dx;
	}
	y += dy;
     }
}

// display matrix
void displayResult(int *results, int height, int width) { 
     gil::rgb8_image_t img(height, width);
     auto img_view = gil::view(img);

     for (int i = 0; i < height; i++) {
	for (int j = 0; j < width; j++) {
          float temp = results[i * width + j] / 512.0;
          printf("%d ", results[i*width+j]);
	  img_view(j,i) = render(temp);
	}
        printf("\n");
     }
     printf("Print successful\n");
     gil::png_write_view("mandelbrot_cyclic.png", const_view(img));
}

int main(int argc, char* argv[]) {
     int root = 0;   
     double t;
     int job_height, data_size; 
     double minX, maxX, minY, maxY, dx, dy;
     int height, width, size, rank;
     int *result, *results, *permresults;
     minX = -2.1;
     maxX = 0.7;
     minY = -1.25;
     maxY = 1.25;

     if (argc == 3) {
	height = atoi (argv[1]);
        width = atoi (argv[2]);
        assert (height > 0 && width > 0);
     }

     else {
	fprintf (stderr, "usage: %s <height> <width>\n", argv[0]);
	fprintf (stderr, "where <height> and <width> are the dimensions of the image.\n");
	return -1;
     }

     dx = (maxX - minX)/width; // x-coordinate
     dy = (maxY - minY)/height; // y-coordinate
     
     // end general initial, begin MPI initialization

     MPI_Init(&argc, &argv);
     MPI_Comm_size(MPI_COMM_WORLD, &size);
     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
     
     t = MPI_Wtime();
     job_height = height / size; 
     
     data_size = job_height * width;
     result = new int[data_size];
    
     // end MPI initialization, begin cyclic computation
     start(result, minX, maxX, minY, maxY, dx, dy, height, width, job_height, rank, size);

     if (rank == 0) {
	results = new int[data_size * size];
	permresults = new int[height * width];
     } 
 
     double comm_start = MPI_Wtime();
     MPI_Gather(result, data_size, MPI_INT, results, data_size, MPI_INT, root, MPI_COMM_WORLD);

     if (rank == 0) {
        printf("rank: %d; communication time: %f\n", rank, MPI_Wtime() - comm_start);
        printf("\n");
	fixMatrix(results, permresults, size, height, width, job_height); // fix the matrix (since MPI_gather gathers contigulously)
        delete[] results;
        doRemainingRows(permresults, data_size, size, height, width, job_height, minX, minY, dx, dy);  // do the remaining rows at root
        printf("rank: %d; total time (comp + comm): %f\n", rank, MPI_Wtime() - t);     
	displayResult(permresults, height, width);
     }

     // end computation
     MPI_Finalize();
     delete[] result;
     delete[] permresults;
     return 0;
}
