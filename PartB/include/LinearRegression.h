#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include "Matrix.h"
#include "Dataset.h"
#include <vector>

/**
 * @brief Linear Regression class for CPU performance prediction
 * Implements PRP = x1*MYCT + x2*MMIN + x3*MMAX + x4*CACH + x5*CHMIN + x6*CHMAX
 */
class LinearRegression {
private:
    std::vector<double> coefficients;  // Model parameters [x1, x2, x3, x4, x5, x6]
    bool isTrained;
    
    // Statistics
    double trainRMSE;
    double testRMSE;
    double rSquared;

public:
    // Constructor
    LinearRegression();
    
    // Destructor
    ~LinearRegression() = default;

    // Train the model using normal equation: theta = (X^T * X)^(-1) * X^T * y
    bool train(const Dataset& trainData);
    
    // Train with regularization (Ridge regression)
    bool trainWithRegularization(const Dataset& trainData, double lambda = 0.01);
    
    // Predict single value
    double predict(const DataPoint& point) const;
    double predict(const std::vector<double>& features) const;
    
    // Predict multiple values
    std::vector<double> predict(const Dataset& testData) const;
    
    // Evaluate model performance
    double calculateRMSE(const Dataset& testData) const;
    double calculateMSE(const Dataset& testData) const;
    double calculateMAE(const Dataset& testData) const;
    double calculateRSquared(const Dataset& testData) const;
    
    // Get model parameters
    const std::vector<double>& getCoefficients() const { return coefficients; }
    bool getIsTrained() const { return isTrained; }
    
    // Display model information
    void displayModel() const;
    void displayEquation() const;
    
    // Cross-validation
    double crossValidate(const Dataset& data, int folds = 5) const;

private:
    // Helper functions
    Matrix createDesignMatrix(const Dataset& data) const;
    std::vector<double> createTargetVector(const Dataset& data) const;
    double calculateMean(const std::vector<double>& values) const;
};

#endif // LINEAR_REGRESSION_H
