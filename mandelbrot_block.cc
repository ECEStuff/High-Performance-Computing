/**
 *  \file mandelbrot_block.cc
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

     // cy marks offset, e.g. with 4 processors and 800x800, offsets are 0|200|400|600
     for (int i = 0, cy = rank * job_size; i < job_size && cy < height; i++, cy++) { // job_size = height / size
	y = cy * dy + minY;
        x = minX;
        for (int j = 0; j < width; ++j) {
          temp[i * width + j] = mandelbrot(x, y); // was temp[i * width + j]
          x += dx;
        }
     }                                                                                
}

// process 0 gets any leftover rows
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

// display the matrix
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
     gil::png_write_view("mandelbrot_block.png", const_view(img));
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
     
     // MPI initialization begin
     MPI_Init(&argc, &argv);
     MPI_Comm_size(MPI_COMM_WORLD, &size);
     MPI_Comm_rank(MPI_COMM_WORLD, &rank);

     t = MPI_Wtime();
     job_height = height / size;

     data_size = job_height * width;
     result = new int[data_size];
     
     // MPI initialization end; computation begin
     start(result, minX, maxX, minY, maxY, dx, dy, height, width, job_height, rank, size);

     if (rank == 0) {
	results = new int[height * width];
     }
     
     double comm_start = MPI_Wtime();
     MPI_Gather(result, data_size, MPI_INT, results, data_size, MPI_INT, root, MPI_COMM_WORLD);
     
     if (rank == 0) {
        printf("rank: %d; communication time: %f\n", rank, MPI_Wtime() - comm_start);
        printf("\n");
        doRemainingRows(results, data_size, size, height, width, job_height, minX, minY, dx, dy);
        printf("rank: %d; total time (comp + comm): %f\n", rank, MPI_Wtime() - t);
        displayResult(results, height, width);
     }

     // computation end
     MPI_Finalize();
     delete[] result;
     delete[] results;
     return 0;
}

/* eof */
