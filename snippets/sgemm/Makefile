SRCS=$(wildcard *.cc)
OBJS=$(SRCS:%.cc=%.o)
DEPS=$(SRCS:%.cc=%.d)
CXX=clang++

all: main

main: main.cc
	$(CXX) -O3 -march=native -lblis -o $@ $<

test: all
	@./main

clean:
	@rm -rf main *.so

-include $(DEPS)

.PHONY: all test clean
