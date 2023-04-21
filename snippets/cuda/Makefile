# SRCS=$(wildcard *.cc)
# OBJS=$(SRCS:%.cc=%.o)
# DEPS=$(SRCS:%.cc=%.d)
CCFLAGS= 
LDFLAGS= 
CC=clang

all: main

main: main.c libcuda-gemm-act.so
	$(CC) -O3 $(LDFLAGS) -o $@ main.c -L. -lcuda-gemm-act

libcuda-gemm-act.so: cuda-gemm-act.cu
	nvcc -O3 -shared -Xcompiler -fPIC -lcublas -lcurand -o $@ $<

test: all
	@./main

clean:
	@rm -rf main *.so

-include $(DEPS)

.PHONY: all test clean
