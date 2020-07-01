all : magic.c magic_gpu.cu
	nvcc magic.c magic_gpu.cu rawspec_rawutils.c -o magic

.PHONY: clean

clean:
	rm -f *.o *.so *.out magic

