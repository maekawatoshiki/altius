# SRCS=$(wildcard *.cc)
# OBJS=$(SRCS:%.cc=%.o)
# DEPS=$(SRCS:%.cc=%.d)
CCFLAGS= 
LDFLAGS= 
CC=clang

all: main

main: main.c libcuda-gemm-act.so
	$(CC) -O3 $(LDFLAGS) -o $@ main.c -L. -lcuda-gemm-act -lblis

libcuda-gemm-act.so: cuda-gemm-act.cu
	nvcc -O3 -shared -Xcompiler -fPIC -lcublas -lcurand -o $@ $<

format: main.c cuda-gemm-act.cu
	clang-format -i -style=llvm $^

test: all
	@./main

clean:
	@rm -rf main *.so

-include $(DEPS)

.PHONY: all test clean
