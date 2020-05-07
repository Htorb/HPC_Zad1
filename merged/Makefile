CC=/usr/local/cuda/bin/nvcc

gpulouvain: gpulouvain.o utils.o
	$(CC) main.cu gpulouvain.o utils.o -o gpulouvain

gpulouvain.o: gpulouvain.h gpulouvain.cu 
	$(CC) -c gpulouvain.cu 

utils.o: utils.h utils.cu
	$(CC) -c utils.cu 
