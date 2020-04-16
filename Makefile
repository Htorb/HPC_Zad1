CC=g++
CFLAGS=-std=c++17 -Wextra

cpulouvain: cpulouvain.cpp
	$(CC) $(CFLAGS) $^ -o $@

clean:
	rm cpulouvain

