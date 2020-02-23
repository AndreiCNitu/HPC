SRC=lbm.cpp
EXE=lbm

build-2080ti :
	syclcc -std=c++14 -O3 --hipsycl-gpu-arch=sm_75 --hipsycl-platform=cuda $(SRC) -o $(EXE)

build-1080ti :
	syclcc -std=c++14 -O3 --hipsycl-gpu-arch=sm_61 --hipsycl-platform=cuda $(SRC) -o $(EXE)

build-radeonvii :
	syclcc -std=c++14 -O3 --hipsycl-gpu-arch=gfx906 --hipsycl-platform=rocm $(SRC) -o $(EXE)

STATE=./final_state.dat
AV_VELS=./av_vels.dat
STATE_SMALL=check/128x128.final_state.dat
VELS_SMALL=check/128x128.av_vels.dat
STATE_MEDIUM_ODD=check/128x256.final_state.dat
VELS_MEDIUM_ODD=check/128x256.av_vels.dat
STATE_MEDIUM=check/256x256.final_state.dat
VELS_MEDIUM=check/256x256.av_vels.dat
STATE_LARGE=check/1024x1024.final_state.dat
VELS_LARGE=check/1024x1024.av_vels.dat

check-small :
	python check/check.py --ref-av-vels-file=$(VELS_SMALL) --ref-final-state-file=$(STATE_SMALL) --av-vels-file=$(AV_VELS) --final-state-file=$(STATE)
check-medium-odd :
	python check/check.py --ref-av-vels-file=$(VELS_MEDIUM_ODD) --ref-final-state-file=$(STATE_MEDIUM_ODD) --av-vels-file=$(AV_VELS) --final-state-file=$(STATE)
check-medium :
	python check/check.py --ref-av-vels-file=$(VELS_MEDIUM) --ref-final-state-file=$(STATE_MEDIUM) --av-vels-file=$(AV_VELS) --final-state-file=$(STATE)
check-large :
	python check/check.py --ref-av-vels-file=$(VELS_LARGE) --ref-final-state-file=$(STATE_LARGE) --av-vels-file=$(AV_VELS) --final-state-file=$(STATE)

.PHONY: all check clean

clean:
	rm -f lbm
