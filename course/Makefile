NVCC      ?= nvcc
ARCH      ?= -arch=sm_120
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo --ptxas-options=-v -Icommon
LDLIBS    ?=

SRCS := $(shell find 0* -name '*.cu')
BINS := $(SRCS:.cu=)

.PHONY: all clean
all: $(BINS)

%: %.cu common/check.h
	$(NVCC) $(NVCCFLAGS) $(ARCH) $< -o $@ $(LDLIBS)

clean:
	rm -f $(BINS)
