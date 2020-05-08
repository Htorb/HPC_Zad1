CC=/usr/local/cuda/bin/nvcc
OBJECTS=utils.o thrust_wrappers.o 

gpulouvain: $(OBJECTS)
	$(CC) main.cu $(OBJECTS) -o gpulouvain

utils.o: utils.h utils.cu
	$(CC) -c utils.cu 

thrust_wrappers.o : thrust_wrappers.h thrust_wrappers.cu
	$(CC) -c thrust_wrappers.cu 
