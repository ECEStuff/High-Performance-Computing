/**
 *  \file mandelbrot_mw.cc
 *
 *  \brief Implement your parallel mandelbrot set in this file.
 */

#include <iostream>
#include <cstdlib>
#include "render.hh"
#include <mpi.h>

const int ROOT = 0;
const int TAG[] = {0, 1, 2}; // result | data | finish

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

// computation for every slave
void worker(int start, int *result, double minX, double maxX, double minY, double maxY, double dx, double dy, int height, int width, int job_height) {
     double x, y;     

     for (int i = 0, cy = start; i < job_height && cy < height; i++, cy++) {
	y = cy * dy + minY;
	x = minX;
	for (int j = 0; j < width; ++j) {
	  result[i * width + j + 1] = mandelbrot(x, y);
 	  x += dx;
	}
     }
     result[0] = start;

}

// display the matrix
void displayResult(int *results, int height, int width, int size) {
     gil::rgb8_image_t img(height, width);
     auto img_view = gil::view(img);
     int offset = 0; 

     // if root == 1, then we have an offset by 1 (since results[0] would contain the start index) 
     // otherwise the results[] matrix is good to go     
     if (size == 1) {
	offset = 1;
     }

     for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
          float temp = results[i * width + j + offset] / 512.0;
          printf("%d ", results[i * width + j + offset]);
          img_view(j,i) = render(temp);
        }
        printf("\n");
     }
     printf("Print successful\n");
     gil::png_write_view("mandelbrot_mw.png", const_view(img));
}

// update the matrix, using the offset and result from the slaves
void fixMatrix(int offset, int *result, int *results, int width, int job_height) {
     for (int i = 0; i < job_height; i++) {
	for (int j = 0; j < width; j++) {
	  results[offset * width + j] = result[i * width + j + 1];
	}
     }
}

// master process
void master(int *result, double minX, double maxX, double minY, double maxY, double dx, double dy, int height, int width, int job_height, int rank, int size, int data_size, double time) {
     if (size == 1) {
	worker(ROOT, result, minX, maxX, minY, maxY, dx, dy, height, width, job_height);
        printf("rank: %d; total time (comm + comp): %f\n", rank, MPI_Wtime() - time);
	displayResult(result, height, width, size);
        return;
     }

     int *results = new int[height * width];
     MPI_Status stat;
     int actives = 1, jobs = 0;
     
     for (; actives < size && jobs < height; actives++, jobs += job_height) {
	MPI_Send(&jobs, 1, MPI_INT, actives, TAG[1], MPI_COMM_WORLD);
     }

     do {
	MPI_Recv(result, data_size, MPI_INT, MPI_ANY_SOURCE, TAG[0], MPI_COMM_WORLD, &stat); // receive result
	int slave = stat.MPI_SOURCE, offset = result[0]; // offset is the starting height
	actives--;
	
	if (jobs < height) {
	  MPI_Send(&jobs, 1, MPI_INT, slave, TAG[1], MPI_COMM_WORLD); // send "data"
	  jobs += job_height;
	  actives++;	  
	}

  	else {
	  MPI_Send(&jobs, 1, MPI_INT, slave, TAG[2], MPI_COMM_WORLD); // send "finish" otherwise. We need to stop processes if they're done.
	}

        fixMatrix(offset, result, results, width, job_height); // append to results
     } while (actives > 1);

     printf("rank: %d; total time (comm + comp): %f\n", rank, MPI_Wtime() - time);
     displayResult(results, height, width, size);
     delete[] results;	
}

// slave process
void slave(int *result, double minX, double maxX, double minY, double maxY, double dx, double dy, int height, int width, int job_height, int rank, int data_size, int *rankList) {
     MPI_Status stat;
     unsigned int offset;

     MPI_Recv(&offset, 1, MPI_INT, ROOT, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
     while (stat.MPI_TAG == TAG[1]) { // MPI_ANY_TAG must be equal to TAG[1], or data tag
	worker(offset, result, minX, maxX, minY, maxY, dx, dy, height, width, job_height);
	rankList[rank] += job_height * width;
        MPI_Send(result, data_size, MPI_INT, ROOT, TAG[0], MPI_COMM_WORLD);
	MPI_Recv(&offset, 1, MPI_INT, ROOT, MPI_ANY_TAG, MPI_COMM_WORLD, &stat); 
     }
}

// start function for master and slaves
void start(int *result, double minX, double maxX, double minY, double maxY, double dx, double dy, int height, int width, int job_height, int rank, int size, int data_size, int *rankList, double time) {
     if (rank == ROOT) {
	master(result, minX, maxX, minY, maxY, dx, dy, height, width, job_height, rank, size, data_size, time);
     }     

     else {
	slave(result, minX, maxX, minY, maxY, dx, dy, height, width, job_height, rank, data_size, rankList);
        //printf("rank %d; load = %d\n", rank, rankList[rank]); 
     }
}

int main (int argc, char* argv[]) { 
     double t;
     int job_height, data_size;
     double minX, maxX, minY, maxY, dx, dy;
     int height, width, size, rank;
     int *result, *rankList; // use *result to store the units of work and *results to build    
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

     dx = (maxX - minX) / width; // x-coord
     dy = (maxY - minY) / height; // y-coord
    
     // begin MPI initialization
     MPI_Init(&argc, &argv);
     MPI_Comm_size(MPI_COMM_WORLD, &size);
     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
     t = MPI_Wtime();
     
     if (size == 1) {
	job_height = height;
     }

     else {
	job_height = 1; // a unit of work shall be 1 row (length of 1 * width)
     }
     
     rankList = new int[size];
   
     for (int i = 0; i < size; i++) {
	rankList[i] = 0;
     }
     
     data_size = job_height * width + 1;
     result = new int[data_size]; // result[0] = position, everything else is data     

     // end MPI initialization, begin computation
     start(result, minX, maxX, minY, maxY, dx, dy, height, width, job_height, rank, size, data_size, rankList, t);
     
     // end computation
     MPI_Finalize();    
     delete[] result;
     delete[] rankList;
}

/* eof */
