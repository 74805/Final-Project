# Variables
CXX = g++
CXXFLAGS = -fdiagnostics-color=always -g -I. \
           -Ithird_party/flatbuffers/include \
           -Ithird_party/gemmlowp \
           -Ithird_party/ruy \
           -Itensorflow/lite/micro/kernels \
           -Itensorflow/lite/c \
           -Itensorflow/lite/micro/tools/make/downloads \
           -Iic \
           -I/usr/include
        #    -Imbed-os \
        #    -Imbed-os/cmsis/CMSIS_5/CMSIS/TARGET_CORTEX_A/Include \
        #    -Imbed-os/platform/include \
        #    -Imbed-os/platform/include/platform \
        #    -Imbed-os/rtos/include \
        
LDFLAGS = -lstdc++

# Source files
CPP_SRCS = $(shell find . -name '*.cpp')
CC_SRCS = $(shell find . -name '*.cc')

# Object files
OBJS = $(CPP_SRCS:.cpp=.o) $(CC_SRCS:.cc=.o)

# Executable
EXEC = main

# Default target
all: $(EXEC)

# Link
$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Compile
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Clean
clean:
	@echo "Object files to be removed: $(OBJS)"
	rm -f $(OBJS) $(EXEC)
