stencil: stencil.cpp
	syclcc -std=c++14 -O3 --hipsycl-gpu-arch=sm_75 --hipsycl-platform=cuda $^ -o $@

.PHONY: all check clean

clean:
	rm -f stencil
