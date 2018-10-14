stencil: stencil3.c
	icc -std=c99 -O1 -Wall $^ -o $@
