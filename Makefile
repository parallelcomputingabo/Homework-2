.PHONY: clean

override CFLAGS := -Wall -Wextra -Wpedantic -Werror -fopenmp -O3 $(CFLAGS)

matmul: main.cpp
	g++ $(CFLAGS) -o matmul main.cpp

matmul_p: main.cpp
	g++ $(CFLAGS) -DPERFORMANCE_MD -o matmul_p main.cpp

clean:
	rm -f ./matmul
