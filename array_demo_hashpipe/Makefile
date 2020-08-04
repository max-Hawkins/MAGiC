CC          = gcc
HPDEMO_LIB_CCFLAGS     = -g -O3 -fPIC -shared -lstdc++ -mavx -msse4 \
                     -I. -I$(CUDA_DIR)/include -I/usr/local/include \
                     -L. -L/usr/local/lib \
                     -lhashpipe -lrt -lm
HPDEMO_LIB_TARGET   = array_demo_hashpipe.so
HPDEMO_LIB_SOURCES  = array_demo_net_thread.c \
                      array_demo_databuf.c
HPDEMO_LIB_INCLUDES = array_demo_databuf.h

all: $(HPDEMO_LIB_TARGET)

$(HPDEMO_LIB_TARGET): $(HPDEMO_LIB_SOURCES) $(HPDEMO_LIB_INCLUDES)
	$(CC) -o $(HPDEMO_LIB_TARGET) $(HPDEMO_LIB_SOURCES) $(HPDEMO_LIB_CCFLAGS)

tags:
	ctags -R .
clean:
	rm -f $(HPDEMO_LIB_TARGET) tags

prefix=/usr/local
LIBDIR=$(prefix)/lib
BINDIR=$(prefix)/bin
install-lib: $(HPDEMO_LIB_TARGET)
	mkdir -p "$(DESTDIR)$(LIBDIR)"
	install -p $^ "$(DESTDIR)$(LIBDIR)"
install: install-lib

.PHONY: all tags clean install install-lib
# vi: set ts=8 noet :
