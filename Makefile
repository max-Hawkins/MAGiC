all : magic.c magic_gpu.cu
	nvcc magic.c magic_gpu.cu rawspec_rawutils.c -o magic

# all: magic_cuda.o rawspec_hdr.o magic

# magic_cuda.o: magic_gpu.cu
# 	nvcc -c magic_gpu.cu -o magic_cuda.o

# rawspec_hdr.o: rawspec_rawutils.h rawspec_rawutils.c
# 	gcc -c -o rawspec_hdr.o rawspec_rawutils.c -o rawspec_hdr.o

# magic: magic_cuda.o rawspec_hdr.o magic.c magic.h
# 	gcc -o magic magic_cuda.o rawspec_hdr.o magic.c

.PHONY: clean

clean:
	rm -f *.o *.so *.out magic

