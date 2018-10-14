stencil: stencil5.c
	icc -std=c99 -march=native -O2 -simd -qopt-report=5 -qopt-report-phase=vec -Wall $^ -o $@
