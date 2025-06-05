#include "../include/LinearRegression.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <numeric>

// Constructor
LinearRegression::LinearRegression() 
    : coefficients(6, 0.0), isTrained(false), trainRMSE(0.0), testRMSE(0.0), rSquared(0.0) {}

// Train the model using normal equation
bool LinearRegression::train(const Dataset& trainData) {
    if (trainData.empty()) {
        std::cerr << "Error: Training dataset is empty" << std::endl;
        return false;
    }

    try {
        // Create design matrix X and target vector y
        Matrix X = createDesignMatrix(trainData);
        std::vector<double> y_vec = createTargetVector(trainData);
        
        // Convert y vector to matrix for calculations
        Matrix y(y_vec.size(), 1);
        for (size_t i = 0; i < y_vec.size(); ++i) {
            y(i, 0) = y_vec[i];
        }

        std::cout << "Design matrix X dimensions: " << X.getRows() << "x" << X.getCols() << std::endl;
        std::cout << "Target vector y dimensions: " << y.getRows() << "x" << y.getCols() << std::endl;

        // Normal equation: theta = (X^T * X)^(-1) * X^T * y
        Matrix Xt = X.transpose();
        Matrix XtX = Xt * X;
        
        std::cout << "Computing matrix inverse..." << std::endl;
        Matrix XtX_inv = XtX.inverse();
        Matrix Xty = Xt * y;
        Matrix theta = XtX_inv * Xty;

        // Extract coefficients
        coefficients.clear();
        coefficients.resize(6);
        for (size_t i = 0; i < 6; ++i) {
            coefficients[i] = theta(i, 0);
        }

        isTrained = true;
        
        // Calculate training RMSE
        trainRMSE = calculateRMSE(trainData);
        
        std::cout << "Model training completed successfully!" << std::endl;
        std::cout << "Training RMSE: " << trainRMSE << std::endl;
        
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error during training: " << e.what() << std::endl;
        return false;
    }
}

// Train with regularization (Ridge regression)
bool LinearRegression::trainWithRegularization(const Dataset& trainData, double lambda) {
    if (trainData.empty()) {
        std::cerr << "Error: Training dataset is empty" << std::endl;
        return false;
    }

    try {
        // Create design matrix X and target vector y
        Matrix X = createDesignMatrix(trainData);
        std::vector<double> y_vec = createTargetVector(trainData);
        
        Matrix y(y_vec.size(), 1);
        for (size_t i = 0; i < y_vec.size(); ++i) {
            y(i, 0) = y_vec[i];
        }

        // Ridge regression: theta = (X^T * X + lambda * I)^(-1) * X^T * y
        Matrix Xt = X.transpose();
        Matrix XtX = Xt * X;
        Matrix I = Matrix::identity(XtX.getRows());
        Matrix regularized = XtX + I * lambda;
        
        Matrix regularized_inv = regularized.inverse();
        Matrix Xty = Xt * y;
        Matrix theta = regularized_inv * Xty;

        // Extract coefficients
        coefficients.clear();
        coefficients.resize(6);
        for (size_t i = 0; i < 6; ++i) {
            coefficients[i] = theta(i, 0);
        }

        isTrained = true;
        trainRMSE = calculateRMSE(trainData);
        
        std::cout << "Ridge regression training completed successfully!" << std::endl;
        std::cout << "Lambda: " << lambda << ", Training RMSE: " << trainRMSE << std::endl;
        
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error during ridge regression training: " << e.what() << std::endl;
        return false;
    }
}

// Predict single value from DataPoint
double LinearRegression::predict(const DataPoint& point) const {
    if (!isTrained) {
        throw std::runtime_error("Model has not been trained yet");
    }
    
    return predict(point.getFeatureVector());
}

// Predict single value from feature vector
double LinearRegression::predict(const std::vector<double>& features) const {
    if (!isTrained) {
        throw std::runtime_error("Model has not been trained yet");
    }
    
    if (features.size() != 6) {
        throw std::invalid_argument("Feature vector must have exactly 6 elements");
    }

    double prediction = 0.0;
    for (size_t i = 0; i < 6; ++i) {
        prediction += coefficients[i] * features[i];
    }
    
    return prediction;
}

// Predict multiple values
std::vector<double> LinearRegression::predict(const Dataset& testData) const {
    std::vector<double> predictions;
    predictions.reserve(testData.size());
    
    for (size_t i = 0; i < testData.size(); ++i) {
        predictions.push_back(predict(testData[i]));
    }
    
    return predictions;
}

// Calculate Root Mean Square Error
double LinearRegression::calculateRMSE(const Dataset& testData) const {
    if (!isTrained) {
        throw std::runtime_error("Model has not been trained yet");
    }
    
    double sumSquaredErrors = 0.0;
    size_t n = testData.size();
    
    for (size_t i = 0; i < n; ++i) {
        double prediction = predict(testData[i]);
        double actual = testData[i].getTarget();
        double error = prediction - actual;
        sumSquaredErrors += error * error;
    }
    
    return std::sqrt(sumSquaredErrors / n);
}

// Calculate Mean Square Error
double LinearRegression::calculateMSE(const Dataset& testData) const {
    if (!isTrained) {
        throw std::runtime_error("Model has not been trained yet");
    }
    
    double sumSquaredErrors = 0.0;
    size_t n = testData.size();
    
    for (size_t i = 0; i < n; ++i) {
        double prediction = predict(testData[i]);
        double actual = testData[i].getTarget();
        double error = prediction - actual;
        sumSquaredErrors += error * error;
    }
    
    return sumSquaredErrors / n;
}

// Calculate Mean Absolute Error
double LinearRegression::calculateMAE(const Dataset& testData) const {
    if (!isTrained) {
        throw std::runtime_error("Model has not been trained yet");
    }
    
    double sumAbsoluteErrors = 0.0;
    size_t n = testData.size();
    
    for (size_t i = 0; i < n; ++i) {
        double prediction = predict(testData[i]);
        double actual = testData[i].getTarget();
        sumAbsoluteErrors += std::abs(prediction - actual);
    }
    
    return sumAbsoluteErrors / n;
}

// Calculate R-squared
double LinearRegression::calculateRSquared(const Dataset& testData) const {
    if (!isTrained) {
        throw std::runtime_error("Model has not been trained yet");
    }
    
    // Calculate mean of actual values
    double meanActual = 0.0;
    size_t n = testData.size();
    
    for (size_t i = 0; i < n; ++i) {
        meanActual += testData[i].getTarget();
    }
    meanActual /= n;
    
    // Calculate sum of squares
    double totalSumSquares = 0.0;  // TSS
    double residualSumSquares = 0.0;  // RSS
    
    for (size_t i = 0; i < n; ++i) {
        double actual = testData[i].getTarget();
        double prediction = predict(testData[i]);
        
        totalSumSquares += (actual - meanActual) * (actual - meanActual);
        residualSumSquares += (actual - prediction) * (actual - prediction);
    }
    
    // R² = 1 - (RSS / TSS)
    if (totalSumSquares == 0.0) {
        return 1.0;  // Perfect prediction if no variance in actual values
    }
    
    return 1.0 - (residualSumSquares / totalSumSquares);
}

// Display model information
void LinearRegression::displayModel() const {
    std::cout << "\n=== Linear Regression Model ===" << std::endl;
    
    if (!isTrained) {
        std::cout << "Model has not been trained yet." << std::endl;
        return;
    }
    
    std::cout << "Model Status: Trained" << std::endl;
    std::cout << "Training RMSE: " << std::fixed << std::setprecision(4) << trainRMSE << std::endl;
    
    std::cout << "\nModel Coefficients:" << std::endl;
    std::vector<std::string> featureNames = {"MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX"};
    
    for (size_t i = 0; i < coefficients.size(); ++i) {
        std::cout << "  " << featureNames[i] << ": " 
                  << std::setw(12) << std::fixed << std::setprecision(6) 
                  << coefficients[i] << std::endl;
    }
}

// Display equation
void LinearRegression::displayEquation() const {
    if (!isTrained) {
        std::cout << "Model has not been trained yet." << std::endl;
        return;
    }
    
    std::cout << "\n=== Linear Regression Equation ===" << std::endl;
    std::cout << "PRP = ";
    
    std::vector<std::string> featureNames = {"MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX"};
    
    for (size_t i = 0; i < coefficients.size(); ++i) {
        if (i > 0) {
            std::cout << (coefficients[i] >= 0 ? " + " : " - ");
        }
        std::cout << std::fixed << std::setprecision(6) << std::abs(coefficients[i]) 
                  << "*" << featureNames[i];
    }
    std::cout << std::endl;
}

// Cross-validation
double LinearRegression::crossValidate(const Dataset& data, int folds) const {
    if (data.size() < static_cast<size_t>(folds)) {
        throw std::invalid_argument("Number of folds cannot be greater than dataset size");
    }
    
    std::vector<double> foldRMSEs;
    size_t foldSize = data.size() / folds;
    
    for (int fold = 0; fold < folds; ++fold) {
        // Create temporary training and validation sets
        Dataset trainSet, validSet;
        
        for (size_t i = 0; i < data.size(); ++i) {
            if (i >= fold * foldSize && i < (fold + 1) * foldSize && fold < folds - 1) {
                validSet.addDataPoint(data[i]);
            } else if (fold == folds - 1 && i >= fold * foldSize) {
                validSet.addDataPoint(data[i]);
            } else {
                trainSet.addDataPoint(data[i]);
            }
        }
        
        // Train temporary model
        LinearRegression tempModel;
        if (tempModel.train(trainSet)) {
            double foldRMSE = tempModel.calculateRMSE(validSet);
            foldRMSEs.push_back(foldRMSE);
        }
    }
    
    // Calculate average RMSE
    if (foldRMSEs.empty()) {
        return -1.0;  // Error indicator
    }
    
    double avgRMSE = std::accumulate(foldRMSEs.begin(), foldRMSEs.end(), 0.0) / foldRMSEs.size();
    
    std::cout << "Cross-validation results (" << folds << " folds):" << std::endl;
    for (int i = 0; i < static_cast<int>(foldRMSEs.size()); ++i) {
        std::cout << "  Fold " << (i + 1) << " RMSE: " << foldRMSEs[i] << std::endl;
    }
    std::cout << "  Average RMSE: " << avgRMSE << std::endl;
    
    return avgRMSE;
}

// Create design matrix from dataset
Matrix LinearRegression::createDesignMatrix(const Dataset& data) const {
    size_t n = data.size();
    Matrix X(n, 6);  // 6 features: MYCT, MMIN, MMAX, CACH, CHMIN, CHMAX
    
    for (size_t i = 0; i < n; ++i) {
        std::vector<double> features = data[i].getFeatureVector();
        for (size_t j = 0; j < 6; ++j) {
            X(i, j) = features[j];
        }
    }
    
    return X;
}

// Create target vector from dataset
std::vector<double> LinearRegression::createTargetVector(const Dataset& data) const {
    std::vector<double> y;
    y.reserve(data.size());
    
    for (size_t i = 0; i < data.size(); ++i) {
        y.push_back(data[i].getTarget());
    }
    
    return y;
}

// Calculate mean of a vector
double LinearRegression::calculateMean(const std::vector<double>& values) const {
    if (values.empty()) {
        return 0.0;
    }
    
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}
