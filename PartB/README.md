# CPU Performance Linear Regression Predictor

A comprehensive C++ implementation of linear regression for predicting CPU relative performance based on hardware specifications. This project uses Object-Oriented Programming principles and implements the mathematical foundations from scratch.

## Overview

This project implements a linear regression model to predict CPU relative performance using the Computer Hardware dataset from the UCI Machine Learning Repository. The model follows the equation:

```
PRP = x1*MYCT + x2*MMIN + x3*MMAX + x4*CACH + x5*CHMIN + x6*CHMAX
```

Where:

- **MYCT**: Machine cycle time in nanoseconds
- **MMIN**: Minimum main memory in kilobytes
- **MMAX**: Maximum main memory in kilobytes
- **CACH**: Cache memory in kilobytes
- **CHMIN**: Minimum channels in units
- **CHMAX**: Maximum channels in units
- **PRP**: Published relative performance (target variable)

## Features

### Core Functionality

- **Linear Regression**: Normal equation implementation with matrix operations
- **Ridge Regression**: Regularized linear regression to prevent overfitting
- **Cross-Validation**: K-fold cross-validation for model validation
- **Comprehensive Evaluation**: RMSE, MSE, MAE, R-squared, MAPE metrics

### Mathematical Components

- **Matrix Class**: Full implementation with operations (multiplication, transpose, inverse)
- **Gaussian Elimination**: Matrix inversion using partial pivoting
- **Statistical Analysis**: Residual analysis and performance metrics

### Data Handling

- **CSV Parser**: Robust data loading with error handling
- **Train/Test Split**: Automatic dataset splitting (80/20 default)
- **Data Validation**: Input validation and preprocessing

## Prerequisites

Ensure you have the following tools installed and configured:

- A C++17 compatible compiler (e.g., g++, clang, or MSVC)
- CMake (version 3.10 or later)
- Make (GNU Make) if using the Makefile
- PowerShell (for build.ps1 script) with execution policy set to allow script execution
- Git (optional, for version control)

## Project Structure

```
Project/
├── main.cpp                 # Main application with interactive menu
├── Makefile                 # Build configuration for Make
├── CMakeLists.txt           # Build configuration for CMake
├── README.md                # This file
├── Data/
│   ├── machine.data         # CPU performance dataset
│   └── machine.names        # Dataset description
├── include/                 # Header files
│   ├── DataPoint.h          # Single data point representation
│   ├── Dataset.h            # Dataset management class
│   ├── LinearRegression.h   # Linear regression implementation
│   ├── Matrix.h             # Matrix operations class
│   └── Evaluator.h          # Model evaluation utilities
└── src/                     # Source files
    ├── DataPoint.cpp
    ├── Dataset.cpp
    ├── LinearRegression.cpp
    ├── Matrix.cpp
    └── Evaluator.cpp
```

## Building the Project

### Option 1: Using Make (Recommended)

```bash
# Build the project
make

# Build and run
make run

# Clean build files
make clean

# Build debug version
make debug

# Show help
make help
```

### Option 1b: Using PowerShell Script (Windows only)

```powershell
# From the project root, run the build script
# Build the project
.\build.ps1 build

# Clean build artifacts
.\build.ps1 clean

# Build and run the program
.\build.ps1 run

# Display help
.\build.ps1 help
```

### Option 1c: Using Command Prompt (Windows)

```cmd
:: From the project root, run the build script
:: Build the project
build.ps1 build

:: Clean build artifacts
build.ps1 clean

:: Build and run the program
build.ps1 run

:: Display help
build.ps1 help
```

### Option 2: Using CMake

```bash
# Create build directory
mkdir build
cd build

# Configure and build
cmake ..
cmake --build .

# Run the program
cd ..
./build/bin/cpu_performance_predictor
```

### Option 3: Manual Compilation

```bash
# Create necessary directories
mkdir -p obj bin

# Compile source files
g++ -std=c++17 -Wall -Wextra -O2 -Iinclude -c src/DataPoint.cpp -o obj/DataPoint.o
g++ -std=c++17 -Wall -Wextra -O2 -Iinclude -c src/Matrix.cpp -o obj/Matrix.o
g++ -std=c++17 -Wall -Wextra -O2 -Iinclude -c src/Dataset.cpp -o obj/Dataset.o
g++ -std=c++17 -Wall -Wextra -O2 -Iinclude -c src/LinearRegression.cpp -o obj/LinearRegression.o
g++ -std=c++17 -Wall -Wextra -O2 -Iinclude -c src/Evaluator.cpp -o obj/Evaluator.o
g++ -std=c++17 -Wall -Wextra -O2 -Iinclude -c main.cpp -o obj/main.o

# Link executable
g++ -std=c++17 -Wall -Wextra -O2 obj/*.o -o bin/cpu_performance_predictor
```

## Usage

### Interactive Menu

Run the program to access the interactive menu:

```bash
./bin/cpu_performance_predictor
```

The menu provides the following options:

1. **Load Dataset**: Load and display dataset statistics
2. **Train Model**: Train linear regression using normal equation
3. **Ridge Regression**: Train with regularization parameter
4. **Evaluate Model**: Test model performance on test set
5. **Individual Prediction**: Make predictions for custom hardware specs
6. **Cross-Validation**: Perform k-fold cross-validation
7. **Detailed Report**: Generate comprehensive evaluation report
8. **Model Equation**: Display the learned equation
9. **Residual Analysis**: Analyze prediction residuals

### Example Workflow

1. Start by loading the dataset (Option 1)
2. Train the model (Option 2 or 3)
3. Evaluate performance (Option 4)
4. Make individual predictions (Option 5)
5. Generate detailed report (Option 7)

## Classes Overview

### DataPoint

Represents a single CPU specification with all features and target value.

```cpp
DataPoint point("ibm", "3033", 57, 4000, 16000, 1, 6, 12, 132, 82);
std::vector<double> features = point.getFeatureVector();
double target = point.getTarget();
```

### Matrix

Comprehensive matrix operations for linear algebra computations.

```cpp
Matrix A(3, 3);
Matrix B = A.transpose();
Matrix C = A.inverse();
Matrix D = A * B;
```

### Dataset

Manages data loading, splitting, and preprocessing.

```cpp
Dataset dataset;
dataset.loadFromFile("Data/machine.data");
dataset.split(0.8, trainSet, testSet);
```

### LinearRegression

Core regression implementation with normal equation and Ridge regression.

```cpp
LinearRegression model;
model.train(trainSet);
double prediction = model.predict(testPoint);
double rmse = model.calculateRMSE(testSet);
```

### Evaluator

Comprehensive model evaluation and analysis tools.

```cpp
Evaluator evaluator(&model);
auto results = evaluator.evaluate(testSet);
evaluator.generateReport(testSet, "report.txt");
```

## Mathematical Implementation

### Normal Equation

The model parameters are calculated using:

```
θ = (X^T * X)^(-1) * X^T * y
```

### Ridge Regression

For regularization:

```
θ = (X^T * X + λI)^(-1) * X^T * y
```

### Performance Metrics

- **RMSE**: Root Mean Square Error
- **MSE**: Mean Square Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of Determination
- **MAPE**: Mean Absolute Percentage Error

## Dataset Information

- **Source**: UCI Machine Learning Repository
- **Instances**: 209 CPU specifications
- **Features**: 6 predictive attributes
- **Target**: Published relative performance (PRP)
- **Range**: Performance values from 6 to 1238

### Feature Statistics

| Feature | Description     | Min | Max   | Mean    |
| ------- | --------------- | --- | ----- | ------- |
| MYCT    | Cycle time (ns) | 17  | 1500  | 203.8   |
| MMIN    | Min memory (KB) | 64  | 32000 | 2867.8  |
| MMAX    | Max memory (KB) | 64  | 64000 | 11796.8 |
| CACH    | Cache (KB)      | 0   | 256   | 25.2    |
| CHMIN   | Min channels    | 0   | 52    | 4.7     |
| CHMAX   | Max channels    | 0   | 176   | 18.3    |
