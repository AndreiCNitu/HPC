stencil: stencil6.c
	icc -std=c99 -g -simd -march=native -ansi-alias -O2 -qopt-report=5 -qopt-report-phase=vec -ipo -Wall $^ -o $@
	#gcc -pg -std=c99 -O2 -ftree-vectorize -ftree-vectorizer-verbose=2 -Wall $^ -o $@
