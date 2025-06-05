# Makefile for CPU Performance Linear Regression Predictor
# Compiler settings
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2

# Directories
SRCDIR = src
INCDIR = include
OBJDIR = obj
BINDIR = bin

# Create directories if they don't exist
$(shell mkdir -p $(OBJDIR) $(BINDIR))

# Source files
SOURCES = $(wildcard $(SRCDIR)/*.cpp)
OBJECTS = $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)

# Main source file
MAIN_SRC = main.cpp
MAIN_OBJ = $(OBJDIR)/main.o

# Target executable
TARGET = $(BINDIR)/cpu_performance_predictor

# Default target
all: $(TARGET)

# Link the executable
$(TARGET): $(OBJECTS) $(MAIN_OBJ)
	@echo "Linking $@..."
	$(CXX) $(CXXFLAGS) $^ -o $@
	@echo "Build successful! Executable: $@"

# Compile source files
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) -I$(INCDIR) -c $< -o $@

# Compile main file
$(MAIN_OBJ): $(MAIN_SRC)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) -I$(INCDIR) -c $< -o $@

# Clean build files
clean:
	@echo "Cleaning build files..."
	rm -rf $(OBJDIR) $(BINDIR)
	@echo "Clean complete."

# Clean and rebuild
rebuild: clean all

# Run the program
run: $(TARGET)
	@echo "Running the program..."
	cd . && $(TARGET)

# Debug build
debug: CXXFLAGS += -g -DDEBUG
debug: $(TARGET)

# Release build (default)
release: CXXFLAGS += -DNDEBUG
release: $(TARGET)

# Install dependencies (if needed)
install-deps:
	@echo "No external dependencies required for this project."

# Help
help:
	@echo "Available targets:"
	@echo "  all      - Build the program (default)"
	@echo "  clean    - Remove build files"
	@echo "  rebuild  - Clean and build"
	@echo "  run      - Build and run the program"
	@echo "  debug    - Build with debug information"
	@echo "  release  - Build optimized version"
	@echo "  help     - Show this help message"

# Phony targets
.PHONY: all clean rebuild run debug release install-deps help

# Dependencies
$(OBJDIR)/DataPoint.o: $(INCDIR)/DataPoint.h
$(OBJDIR)/Matrix.o: $(INCDIR)/Matrix.h
$(OBJDIR)/Dataset.o: $(INCDIR)/Dataset.h $(INCDIR)/DataPoint.h
$(OBJDIR)/LinearRegression.o: $(INCDIR)/LinearRegression.h $(INCDIR)/Matrix.h $(INCDIR)/Dataset.h
$(OBJDIR)/Evaluator.o: $(INCDIR)/Evaluator.h $(INCDIR)/LinearRegression.h $(INCDIR)/Dataset.h
$(MAIN_OBJ): $(INCDIR)/Dataset.h $(INCDIR)/LinearRegression.h $(INCDIR)/Evaluator.h
