SRCS=$(wildcard *.cc)
OBJS=$(SRCS:%.cc=%.o)
DEPS=$(SRCS:%.cc=%.d)
CXX=clang++

all: main

main: main.cc
	$(CXX) -O3 -fopenmp -march=native -ffp-contract=fast -lm -lblis -lomp -o $@ $<

test: all
	@./main

clean:
	@rm -rf main *.so

-include $(DEPS)

.PHONY: all test clean
