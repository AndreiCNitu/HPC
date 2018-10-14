stencil: stencil3.c
	icc -std=c99 -O2 -Wall $^ -o $@
