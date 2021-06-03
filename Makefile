.DEFAULT_GOAL := all

MPICC = mpic++
MPICFLAGS = -std=c++11
MPICOPTFLAGS = -O3 -g -lpng -Wall
MPILDFLAGS =

TARGETS = mandelbrot_serial$(EXEEXT) mandelbrot_block$(EXEEXT) mandelbrot_cyclic$(EXEEXT) mandelbrot_mw$(EXEEXT)

all: $(TARGETS)

SRCS_COMMON = render.cc 

DISTFILES += $(SRCS_COMMON) $(DEPS_COMMON)

mandelbrot_serial$(EXEEXT): mandelbrot_serial.cc $(SRCS_COMMON) $(DEPS_COMMON)
	$(MPICC) $(MPICFLAGS) $(MPICOPTFLAGS)  \
	    -o $@ mandelbrot_serial.cc $(SRCS_COMMON) $(MPILDFLAGS)

mandelbrot_block$(EXEEXT): mandelbrot_block.cc $(SRCS_COMMON) $(DEPS_COMMON)
	$(MPICC) $(MPICFLAGS) $(MPICOPTFLAGS)  \
	    -o $@ mandelbrot_block.cc $(SRCS_COMMON) $(MPILDFLAGS)

mandelbrot_cyclic$(EXEEXT): mandelbrot_cyclic.cc $(SRCS_COMMON) $(DEPS_COMMON)
	$(MPICC) $(MPICFLAGS) $(MPICOPTFLAGS)  \
	    -o $@ mandelbrot_cyclic.cc $(SRCS_COMMON) $(MPILDFLAGS)

mandelbrot_mw$(EXEEXT): mandelbrot_mw.cc $(SRCS_COMMON) $(DEPS_COMMON)
	$(MPICC) $(MPICFLAGS) $(MPICOPTFLAGS)  \
	    -o $@ mandelbrot_mw.cc $(SRCS_COMMON) $(MPILDFLAGS)

clean:
	rm -f $(TARGETS) 

# eof
