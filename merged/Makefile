CC=/usr/local/cuda/bin/nvcc
DFLAGS=#-g -G 
OBJECTS=utils.o thrust_wrappers.o hashmap_utils.o main.o

gpulouvain: $(OBJECTS) 
	$(CC) $(DFLAGS) $(OBJECTS) -o gpulouvain

main.o:  main.cu
	$(CC) $(DFLAGS) -dc main.cu

utils.o: utils.h utils.cu
	$(CC) $(DFLAGS) -dc utils.cu 

hashmap_utils.o: hashmap_utils.h hashmap_utils.cu
	$(CC) $(DFLAGS) -dc hashmap_utils.cu 

thrust_wrappers.o: thrust_wrappers.h thrust_wrappers.cu
	$(CC) -dc thrust_wrappers.cu 

clean:
	rm *.o gpulouvain