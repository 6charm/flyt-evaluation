# Compiler settings
CC = nvcc
CFLAGS = -O2 -g -arch=sm_60
LDFLAGS =

# Target executable
TARGET = test-rpc-virt-ovh

# Source files and object files
SRC = test-rpc-virt-ovh.cu
OBJ = $(SRC:.cu=.o)

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(OBJ)
	@echo "Linking CUDA object files..."
	$(CC) $(OBJ) -o $(TARGET) $(LDFLAGS)
	@echo "Build complete: $(TARGET)"

# Compile the CUDA source code
%.o: %.cu
	@echo "Compiling CUDA source: $<"
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up build artifacts
clean:
	@echo "Cleaning up..."
	rm -f $(TARGET) $(OBJ)
	@echo "Clean complete."

# Rebuild from scratch
rebuild: clean all

# Phony targets
.PHONY: all clean rebuild

