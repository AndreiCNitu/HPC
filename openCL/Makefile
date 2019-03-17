# Makefile

SRC=d2q9-bgk
EXE=d2q9-bgk

CC=gcc
CFLAGS= -std=c99 -Wall -O3
LIBS = -lm

PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	LIBS += -framework OpenCL
else
	LIBS += -lOpenCL
endif

STATE=./final_state.dat
AV_VELS=./av_vels.dat
STATE_SMALL=check/128x128.final_state.dat
VELS_SMALL=check/128x128.av_vels.dat
STATE_MEDIUM=check/256x256.final_state.dat
VELS_MEDIUM=check/256x256.av_vels.dat
STATE_MEDIUM2=check/128x256.final_state.dat
VELS_MEDIUM2=check/128x256.av_vels.dat
STATE_LARGE=check/1024x1024.final_state.dat
VELS_LARGE=check/1024x1024.av_vels.dat

all: $(EXE)

$(EXE): $(EXE).c
	$(CC) $(CFLAGS) $^ $(LIBS) -o $(EXE)

check-small:
	python check/check.py --ref-av-vels-file=$(VELS_SMALL) --ref-final-state-file=$(STATE_SMALL) --av-vels-file=$(AV_VELS) --final-state-file=$(STATE)
check-medium:
	python check/check.py --ref-av-vels-file=$(VELS_MEDIUM) --ref-final-state-file=$(STATE_MEDIUM) --av-vels-file=$(AV_VELS) --final-state-file=$(STATE)
check-medium-2:
	python check/check.py --ref-av-vels-file=$(VELS_MEDIUM2) --ref-final-state-file=$(STATE_MEDIUM2) --av-vels-file=$(AV_VELS) --final-state-file=$(STATE)
check-large:
	python check/check.py --ref-av-vels-file=$(VELS_LARGE) --ref-final-state-file=$(STATE_LARGE) --av-vels-file=$(AV_VELS) --final-state-file=$(STATE)

.PHONY: all check clean

clean:
	rm -f $(EXE)
	rm -f *.out
	rm -f *.optrpt
