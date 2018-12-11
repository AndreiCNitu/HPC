stencil: stencil.c
	mpiicc -g -O3 -qopt-report=5 -xAVX -fp-model fast=2 -ansi-alias -std=c99 -Wall $^ -o $@

EXES=stencil

clean:
	\rm -f $(EXES) 
	\rm -f *.out
	\rm -f OUT
	\rm -f *.optrpt
	\rm -f machine.*
	\rm -f MPI.*

check-small:
	python check.py --ref-stencil-file stencil_1024_1024_100.pgm --stencil-file stencil.pgm

check-medium:
	python check.py --ref-stencil-file stencil_4096_4096_100.pgm --stencil-file stencil.pgm

check-large:
	python check.py --ref-stencil-file stencil_8000_8000_100.pgm --stencil-file stencil.pgm
