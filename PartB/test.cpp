#include "include/Dataset.h"
#include "include/LinearRegression.h"
#include "include/Evaluator.h"
#include <iostream>
#include <iomanip>

/**
 * @brief Simple test program to validate the linear regression implementation
 */

void testMatrixOperations() {
    std::cout << "=== Testing Matrix Operations ===" << std::endl;
    
    // Test basic matrix operations
    Matrix A(2, 2);
    A(0, 0) = 1; A(0, 1) = 2;
    A(1, 0) = 3; A(1, 1) = 4;
    
    std::cout << "Matrix A:" << std::endl;
    A.display();
    
    Matrix B = A.transpose();
    std::cout << "Matrix A transpose:" << std::endl;
    B.display();
    
    Matrix C = A * B;
    std::cout << "A * A^T:" << std::endl;
    C.display();
    
    // Test matrix inverse
    Matrix D(2, 2);
    D(0, 0) = 4; D(0, 1) = 7;
    D(1, 0) = 2; D(1, 1) = 6;
    
    std::cout << "Matrix D:" << std::endl;
    D.display();
    
    try {
        Matrix D_inv = D.inverse();
        std::cout << "Matrix D inverse:" << std::endl;
        D_inv.display();
        
        Matrix Identity = D * D_inv;
        std::cout << "D * D^(-1) (should be identity):" << std::endl;
        Identity.display();
    }
    catch (const std::exception& e) {
        std::cout << "Error computing inverse: " << e.what() << std::endl;
    }
    
    std::cout << std::endl;
}

void testDatasetLoading() {
    std::cout << "=== Testing Dataset Loading ===" << std::endl;
    
    Dataset dataset;
    if (dataset.loadFromFile("Data/machine.data")) {
        std::cout << "Dataset loaded successfully!" << std::endl;
        std::cout << "Number of samples: " << dataset.size() << std::endl;
        
        if (dataset.size() > 0) {
            std::cout << "First data point:" << std::endl;
            dataset[0].display();
        }
    } else {
        std::cout << "Failed to load dataset!" << std::endl;
    }
    
    std::cout << std::endl;
}

void testLinearRegression() {
    std::cout << "=== Testing Linear Regression ===" << std::endl;
    
    Dataset fullDataset;
    if (!fullDataset.loadFromFile("Data/machine.data")) {
        std::cout << "Failed to load dataset for regression test!" << std::endl;
        return;
    }
    
    // Split dataset
    Dataset trainDataset, testDataset;
    fullDataset.split(0.8, trainDataset, testDataset);
    
    std::cout << "Training samples: " << trainDataset.size() << std::endl;
    std::cout << "Test samples: " << testDataset.size() << std::endl;
    
    // Train model
    LinearRegression model;
    if (model.train(trainDataset)) {
        std::cout << "Model trained successfully!" << std::endl;
        
        // Display model
        model.displayModel();
        
        // Evaluate on test set
        double rmse = model.calculateRMSE(testDataset);
        double r2 = model.calculateRSquared(testDataset);
        
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Test RMSE: " << rmse << std::endl;
        std::cout << "Test R²: " << r2 << std::endl;
        
        // Test individual prediction
        if (testDataset.size() > 0) {
            double prediction = model.predict(testDataset[0]);
            double actual = testDataset[0].getTarget();
            std::cout << "Sample prediction: " << prediction 
                      << " (actual: " << actual 
                      << ", error: " << std::abs(prediction - actual) << ")" << std::endl;
        }
    } else {
        std::cout << "Model training failed!" << std::endl;
    }
    
    std::cout << std::endl;
}

int main() {
    std::cout << "CPU Performance Predictor - Test Suite" << std::endl;
    std::cout << "=======================================" << std::endl << std::endl;
    
    try {
        testMatrixOperations();
        testDatasetLoading();
        testLinearRegression();
        
        std::cout << "All tests completed!" << std::endl;
    }
    catch (const std::exception& e) {
        std::cout << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
