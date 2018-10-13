stencil: stencil.c
	icc -pg -std=c99 -O2 -Wall $^ -o $@

