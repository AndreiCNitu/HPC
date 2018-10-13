stencil: stencil4.c
	icc -pg -std=c99 -march=native -O3 -vec_report -simd -Wall $^ -o $@

