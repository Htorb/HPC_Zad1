CC=/usr/local/cuda/bin/nvcc
DFLAGS=#-g -G
OFLAGS=-Xptxas -O3#todo report how it improves speed (or not)
OBJECTS=utils.o thrust_wrappers.o hashmap_utils.o main.o
ARCH=-gencode=arch=compute_70,code=sm_70


gpulouvain: $(OBJECTS) 
	$(CC) $(ARCH) -gencode=arch=compute_70,code=compute_70 $(OBJECTS) -o gpulouvain

main.o:  main.cu
	$(CC) $(OFLAGS) $(ARCH) -dc main.cu

utils.o: utils.h utils.cu
	$(CC) $(OFLAGS) $(DFLAGS) $(ARCH) -dc utils.cu 

hashmap_utils.o: hashmap_utils.h hashmap_utils.cu
	$(CC) $(OFLAGS) $(DFLAGS) $(ARCH) -dc hashmap_utils.cu 

thrust_wrappers.o: thrust_wrappers.h thrust_wrappers.cu
	$(CC) $(OFLAGS) $(ARCH) -dc thrust_wrappers.cu 


clean:
	rm *.o gpulouvain