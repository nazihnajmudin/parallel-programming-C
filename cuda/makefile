# Compiler
NVCC = nvcc

# Flags tanpa optimasi
NVCCFLAGS = -std=c++17

# OpenCV flags
OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4)
OPENCV_LIBS = $(shell pkg-config --libs opencv4)

# Direktori output object files
BIN_DIR = bin

# Source files
SOURCES = main.cu image.cu sobel.cu sobel_raw.cu sobel_sm-16x16.cu sobel_sm-dyn.cu optimizer.cu io.cu

# Object files di folder BIN_DIR
OBJECTS = $(patsubst %.cu,$(BIN_DIR)/%.o,$(SOURCES))

# Header directory
HEADER_DIR = header

# Executable name
TARGET = main

# Default target
all: $(TARGET)

# Link all object files to create executable
$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(OPENCV_LIBS)

# Compile CUDA source files with OpenCV flags
$(BIN_DIR)/%.o: %.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -I$(HEADER_DIR) $(OPENCV_CFLAGS) -c $< -o $@

# Pastikan direktori bin ada
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Clean build files
clean:
	rm -f $(BIN_DIR)/*.o $(TARGET)

run:
	./main 0 ../test_cases/fish.jpg fish-0.jpg > output.txt

# Phony targets
.PHONY: all clean run
