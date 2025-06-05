#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>

/**
 * @brief Matrix class for linear algebra operations
 */
class Matrix {
private:
    std::vector<std::vector<double>> data;
    size_t rows;
    size_t cols;

public:
    // Constructors
    Matrix();
    Matrix(size_t rows, size_t cols);
    Matrix(const std::vector<std::vector<double>>& data);

    // Copy constructor and assignment operator
    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);

    // Destructor
    ~Matrix() = default;

    // Getters
    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }
    
    // Element access
    double& operator()(size_t row, size_t col);
    const double& operator()(size_t row, size_t col) const;
    
    // Row access
    std::vector<double>& operator[](size_t row);
    const std::vector<double>& operator[](size_t row) const;

    // Matrix operations
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix operator*(double scalar) const;
    
    // Transpose
    Matrix transpose() const;
    
    // Inverse (using Gaussian elimination)
    Matrix inverse() const;
    
    // Determinant
    double determinant() const;
    
    // Identity matrix
    static Matrix identity(size_t size);
    
    // Zero matrix
    static Matrix zeros(size_t rows, size_t cols);
    
    // Check if matrix is square
    bool isSquare() const { return rows == cols; }
    
    // Display
    void display() const;
    
    // Set element
    void setElement(size_t row, size_t col, double value);
    
    // Resize matrix
    void resize(size_t newRows, size_t newCols);

private:
    // Helper functions for matrix operations
    void swapRows(size_t row1, size_t row2);
    void multiplyRow(size_t row, double factor);
    void addRowMultiple(size_t sourceRow, size_t targetRow, double factor);
};

#endif // MATRIX_H
