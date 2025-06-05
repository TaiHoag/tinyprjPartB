#include "../include/Matrix.h"
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <cmath>

// Default constructor
Matrix::Matrix() : rows(0), cols(0) {}

// Constructor with dimensions
Matrix::Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
    data.resize(rows, std::vector<double>(cols, 0.0));
}

// Constructor from 2D vector
Matrix::Matrix(const std::vector<std::vector<double>>& data) 
    : data(data), rows(data.size()), cols(data.empty() ? 0 : data[0].size()) {}

// Copy constructor
Matrix::Matrix(const Matrix& other) 
    : data(other.data), rows(other.rows), cols(other.cols) {}

// Assignment operator
Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        data = other.data;
        rows = other.rows;
        cols = other.cols;
    }
    return *this;
}

// Element access operators
double& Matrix::operator()(size_t row, size_t col) {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Matrix indices out of range");
    }
    return data[row][col];
}

const double& Matrix::operator()(size_t row, size_t col) const {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Matrix indices out of range");
    }
    return data[row][col];
}

// Row access operators
std::vector<double>& Matrix::operator[](size_t row) {
    if (row >= rows) {
        throw std::out_of_range("Matrix row index out of range");
    }
    return data[row];
}

const std::vector<double>& Matrix::operator[](size_t row) const {
    if (row >= rows) {
        throw std::out_of_range("Matrix row index out of range");
    }
    return data[row];
}

// Matrix addition
Matrix Matrix::operator+(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }
    
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = data[i][j] + other(i, j);
        }
    }
    return result;
}

// Matrix subtraction
Matrix Matrix::operator-(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }
    
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = data[i][j] - other(i, j);
        }
    }
    return result;
}

// Matrix multiplication
Matrix Matrix::operator*(const Matrix& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }
    
    Matrix result(rows, other.cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < other.cols; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < cols; ++k) {
                sum += data[i][k] * other(k, j);
            }
            result(i, j) = sum;
        }
    }
    return result;
}

// Scalar multiplication
Matrix Matrix::operator*(double scalar) const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = data[i][j] * scalar;
        }
    }
    return result;
}

// Transpose
Matrix Matrix::transpose() const {
    Matrix result(cols, rows);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(j, i) = data[i][j];
        }
    }
    return result;
}

// Inverse using Gaussian elimination with partial pivoting
Matrix Matrix::inverse() const {
    if (!isSquare()) {
        throw std::invalid_argument("Matrix must be square to compute inverse");
    }
    
    const double EPSILON = 1e-10;
    size_t n = rows;
    
    // Create augmented matrix [A|I]
    Matrix augmented(n, 2 * n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            augmented(i, j) = data[i][j];
        }
        augmented(i, i + n) = 1.0;  // Identity matrix on the right
    }
    
    // Forward elimination with partial pivoting
    for (size_t i = 0; i < n; ++i) {
        // Find pivot
        size_t maxRow = i;
        for (size_t k = i + 1; k < n; ++k) {
            if (std::abs(augmented(k, i)) > std::abs(augmented(maxRow, i))) {
                maxRow = k;
            }
        }
        
        // Check for singular matrix
        if (std::abs(augmented(maxRow, i)) < EPSILON) {
            throw std::runtime_error("Matrix is singular and cannot be inverted");
        }
        
        // Swap rows if needed
        if (maxRow != i) {
            augmented.swapRows(i, maxRow);
        }
        
        // Make diagonal element 1
        double pivot = augmented(i, i);
        augmented.multiplyRow(i, 1.0 / pivot);
        
        // Eliminate column
        for (size_t k = 0; k < n; ++k) {
            if (k != i) {
                double factor = augmented(k, i);
                augmented.addRowMultiple(i, k, -factor);
            }
        }
    }
    
    // Extract the inverse matrix from the right side of augmented matrix
    Matrix result(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            result(i, j) = augmented(i, j + n);
        }
    }
    
    return result;
}

// Determinant
double Matrix::determinant() const {
    if (!isSquare()) {
        throw std::invalid_argument("Matrix must be square to compute determinant");
    }
    
    if (rows == 1) {
        return data[0][0];
    }
    
    if (rows == 2) {
        return data[0][0] * data[1][1] - data[0][1] * data[1][0];
    }
    
    // Use LU decomposition for larger matrices
    Matrix temp(*this);
    double det = 1.0;
    const double EPSILON = 1e-10;
    
    for (size_t i = 0; i < rows; ++i) {
        // Find pivot
        size_t maxRow = i;
        for (size_t k = i + 1; k < rows; ++k) {
            if (std::abs(temp(k, i)) > std::abs(temp(maxRow, i))) {
                maxRow = k;
            }
        }
        
        if (std::abs(temp(maxRow, i)) < EPSILON) {
            return 0.0;  // Singular matrix
        }
        
        if (maxRow != i) {
            temp.swapRows(i, maxRow);
            det *= -1.0;  // Row swap changes sign
        }
        
        det *= temp(i, i);
        
        // Eliminate below diagonal
        for (size_t k = i + 1; k < rows; ++k) {
            double factor = temp(k, i) / temp(i, i);
            temp.addRowMultiple(i, k, -factor);
        }
    }
    
    return det;
}

// Identity matrix
Matrix Matrix::identity(size_t size) {
    Matrix result(size, size);
    for (size_t i = 0; i < size; ++i) {
        result(i, i) = 1.0;
    }
    return result;
}

// Zero matrix
Matrix Matrix::zeros(size_t rows, size_t cols) {
    return Matrix(rows, cols);  // Constructor initializes with zeros
}

// Display matrix
void Matrix::display() const {
    std::cout << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            std::cout << std::setw(12) << data[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

// Set element
void Matrix::setElement(size_t row, size_t col, double value) {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Matrix indices out of range");
    }
    data[row][col] = value;
}

// Resize matrix
void Matrix::resize(size_t newRows, size_t newCols) {
    rows = newRows;
    cols = newCols;
    data.resize(rows);
    for (auto& row : data) {
        row.resize(cols, 0.0);
    }
}

// Helper functions for row operations
void Matrix::swapRows(size_t row1, size_t row2) {
    if (row1 >= rows || row2 >= rows) {
        throw std::out_of_range("Row indices out of range");
    }
    std::swap(data[row1], data[row2]);
}

void Matrix::multiplyRow(size_t row, double factor) {
    if (row >= rows) {
        throw std::out_of_range("Row index out of range");
    }
    for (size_t j = 0; j < cols; ++j) {
        data[row][j] *= factor;
    }
}

void Matrix::addRowMultiple(size_t sourceRow, size_t targetRow, double factor) {
    if (sourceRow >= rows || targetRow >= rows) {
        throw std::out_of_range("Row indices out of range");
    }
    for (size_t j = 0; j < cols; ++j) {
        data[targetRow][j] += factor * data[sourceRow][j];
    }
}
