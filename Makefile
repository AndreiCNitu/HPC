stencil: stencil5.c
	icc -std=c99 -march=native -O2 -simd -qopt-report=5 -qopt-report-phase=vec - ipo -Wall $^ -o $@
	#gcc -std=c99 -O3 -Wall $^ -o $@
