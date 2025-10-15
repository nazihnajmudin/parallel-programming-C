# Compiler
NVCC = nvcc

# Flags tanpa optimasi
NVCCFLAGS = -std=c++11

# OpenCV flags
OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4)
OPENCV_LIBS = $(shell pkg-config --libs opencv4)

# Direktori output object files
BIN_DIR = bin

# Source files
SOURCES = main.cpp image.cpp sobel.cu optimizer.cu

# Object files di folder BIN_DIR
OBJECTS = $(patsubst %.cpp,$(BIN_DIR)/%.o,$(SOURCES))
OBJECTS := $(patsubst %.cu,$(BIN_DIR)/%.o,$(OBJECTS))

# Header directory
HEADER_DIR = header

# Executable name
TARGET = main

# Default target
all: $(TARGET)

# Link all object files to create executable
$(TARGET): $(OBJECTS)
	$(NVCC) $^ -o $@ $(OPENCV_LIBS)

# Compile C++ files dengan OpenCV compile flags
$(BIN_DIR)/%.o: %.cpp
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -I$(HEADER_DIR) $(OPENCV_CFLAGS) -c $< -o $@

# Compile CUDA files dengan OpenCV compile flags
$(BIN_DIR)/%.o: %.cu
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -I$(HEADER_DIR) $(OPENCV_CFLAGS) -c $< -o $@

# Clean build files
clean:
	rm -f $(BIN_DIR)/*.o $(TARGET)

# Phony targets
.PHONY: all clean