stencil: stencil.cpp
	syclcc -std=c++14 -O3 --hipsycl-gpu-arch=sm_75 --hipsycl-platform=cuda $^ -o $@

.PHONY: all check clean

check-small:
	python check.py --verbose --ref-stencil-file stencil_1024_1024_100.pgm --stencil-file stencil.pgm

check-medium:
	python check.py --verbose --ref-stencil-file stencil_4096_4096_100.pgm --stencil-file stencil.pgm

check-large:
	python check.py --verbose --ref-stencil-file stencil_8000_8000_100.pgm --stencil-file stencil.pgm

clean:
	rm -f stencil
