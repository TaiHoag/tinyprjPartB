#include "../include/Evaluator.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <numeric>

// Constructor
Evaluator::Evaluator(LinearRegression* model) : model(model) {
    if (!model) {
        throw std::invalid_argument("Model pointer cannot be null");
    }
}

// Comprehensive evaluation
Evaluator::EvaluationResults Evaluator::evaluate(const Dataset& testData) const {
    if (!model->getIsTrained()) {
        throw std::runtime_error("Model has not been trained yet");
    }
    
    EvaluationResults results;
    
    // Get predictions and actual values
    results.predictions = model->predict(testData);
    results.actuals.reserve(testData.size());
    
    for (size_t i = 0; i < testData.size(); ++i) {
        results.actuals.push_back(testData[i].getTarget());
    }
    
    // Calculate residuals
    results.residuals = calculateResiduals(results.actuals, results.predictions);
    
    // Calculate metrics
    results.rmse = model->calculateRMSE(testData);
    results.mse = model->calculateMSE(testData);
    results.mae = model->calculateMAE(testData);
    results.rSquared = model->calculateRSquared(testData);
    results.meanAbsolutePercentageError = calculateMAPE(results.actuals, results.predictions);
    
    return results;
}

// Generate detailed evaluation report
void Evaluator::generateReport(const Dataset& testData, const std::string& filename) const {
    EvaluationResults results = evaluate(testData);
    
    std::ostream* output = &std::cout;
    std::ofstream file;
    
    if (!filename.empty()) {
        file.open(filename);
        if (file.is_open()) {
            output = &file;
        }
    }
    
    *output << "=====================================\n";
    *output << "    LINEAR REGRESSION EVALUATION\n";
    *output << "=====================================\n\n";
    
    // Model equation
    *output << "Model Equation:\n";
    *output << "PRP = ";
    std::vector<std::string> featureNames = {"MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX"};
    const auto& coeffs = model->getCoefficients();
    
    for (size_t i = 0; i < coeffs.size(); ++i) {
        if (i > 0) {
            *output << (coeffs[i] >= 0 ? " + " : " - ");
        }
        *output << std::fixed << std::setprecision(6) << std::abs(coeffs[i]) 
                << "*" << featureNames[i];
    }
    *output << "\n\n";
    
    // Performance metrics
    *output << "Performance Metrics:\n";
    *output << "-------------------\n";
    *output << std::fixed << std::setprecision(4);
    *output << "Root Mean Square Error (RMSE): " << results.rmse << "\n";
    *output << "Mean Square Error (MSE):       " << results.mse << "\n";
    *output << "Mean Absolute Error (MAE):     " << results.mae << "\n";
    *output << "R-squared (R²):                " << results.rSquared << "\n";
    *output << "Mean Absolute Percentage Error: " << results.meanAbsolutePercentageError << "%\n";
    *output << "Number of test samples:        " << testData.size() << "\n\n";
    
    // Residual statistics
    double meanResidual = calculateMean(results.residuals);
    double stdResidual = calculateStandardDeviation(results.residuals);
    double minResidual = *std::min_element(results.residuals.begin(), results.residuals.end());
    double maxResidual = *std::max_element(results.residuals.begin(), results.residuals.end());
    
    *output << "Residual Analysis:\n";
    *output << "----------------\n";
    *output << "Mean residual:     " << meanResidual << "\n";
    *output << "Std residual:      " << stdResidual << "\n";
    *output << "Min residual:      " << minResidual << "\n";
    *output << "Max residual:      " << maxResidual << "\n\n";
    
    // Sample predictions
    *output << "Sample Predictions (First 10):\n";
    *output << "-----------------------------\n";
    *output << std::setw(10) << "Actual" << std::setw(12) << "Predicted" 
            << std::setw(12) << "Residual" << std::setw(12) << "% Error\n";
    *output << std::string(46, '-') << "\n";
    
    size_t sampleSize = std::min(static_cast<size_t>(10), testData.size());
    for (size_t i = 0; i < sampleSize; ++i) {
        double percentError = std::abs(results.residuals[i]) / results.actuals[i] * 100.0;
        *output << std::setw(10) << std::fixed << std::setprecision(2) << results.actuals[i]
                << std::setw(12) << results.predictions[i]
                << std::setw(12) << results.residuals[i]
                << std::setw(11) << percentError << "%\n";
    }
    
    if (file.is_open()) {
        file.close();
        std::cout << "Evaluation report saved to: " << filename << std::endl;
    }
}

// Residual analysis
void Evaluator::residualAnalysis(const Dataset& testData) const {
    EvaluationResults results = evaluate(testData);
    
    std::cout << "\n=== Residual Analysis ===" << std::endl;
    
    // Basic statistics
    double meanResidual = calculateMean(results.residuals);
    double stdResidual = calculateStandardDeviation(results.residuals);
    double minResidual = *std::min_element(results.residuals.begin(), results.residuals.end());
    double maxResidual = *std::max_element(results.residuals.begin(), results.residuals.end());
    
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Mean residual:           " << meanResidual << std::endl;
    std::cout << "Standard deviation:      " << stdResidual << std::endl;
    std::cout << "Minimum residual:        " << minResidual << std::endl;
    std::cout << "Maximum residual:        " << maxResidual << std::endl;
    
    // Count residuals in different ranges
    size_t withinOneStd = 0, withinTwoStd = 0, withinThreeStd = 0;
    
    for (double residual : results.residuals) {
        double absResidual = std::abs(residual);
        if (absResidual <= stdResidual) withinOneStd++;
        if (absResidual <= 2 * stdResidual) withinTwoStd++;
        if (absResidual <= 3 * stdResidual) withinThreeStd++;
    }
    
    double total = static_cast<double>(results.residuals.size());
    std::cout << "\nResidual Distribution:" << std::endl;
    std::cout << "Within 1 std dev:  " << std::setw(6) << withinOneStd 
              << " (" << std::setw(5) << std::setprecision(1) << (withinOneStd/total*100) << "%)" << std::endl;
    std::cout << "Within 2 std dev:  " << std::setw(6) << withinTwoStd 
              << " (" << std::setw(5) << (withinTwoStd/total*100) << "%)" << std::endl;
    std::cout << "Within 3 std dev:  " << std::setw(6) << withinThreeStd 
              << " (" << std::setw(5) << (withinThreeStd/total*100) << "%)" << std::endl;
}

// Prediction vs Actual comparison
void Evaluator::predictionComparison(const Dataset& testData, size_t numSamples) const {
    EvaluationResults results = evaluate(testData);
    
    size_t samplesToShow = std::min(numSamples, testData.size());
    
    std::cout << "\n=== Prediction vs Actual Comparison (" << samplesToShow << " samples) ===" << std::endl;
    std::cout << std::setw(6) << "Index" << std::setw(10) << "Actual" 
              << std::setw(12) << "Predicted" << std::setw(12) << "Error" 
              << std::setw(12) << "% Error" << std::setw(15) << "Vendor" << std::endl;
    std::cout << std::string(75, '-') << std::endl;
    
    for (size_t i = 0; i < samplesToShow; ++i) {
        double percentError = std::abs(results.residuals[i]) / results.actuals[i] * 100.0;
        
        std::cout << std::setw(6) << i
                  << std::setw(10) << std::fixed << std::setprecision(2) << results.actuals[i]
                  << std::setw(12) << results.predictions[i]
                  << std::setw(12) << results.residuals[i]
                  << std::setw(11) << percentError << "%"
                  << std::setw(15) << testData[i].getVendor() << std::endl;
    }
}

// Calculate Mean Absolute Percentage Error
double Evaluator::calculateMAPE(const std::vector<double>& actual, const std::vector<double>& predicted) {
    if (actual.size() != predicted.size() || actual.empty()) {
        throw std::invalid_argument("Vectors must be non-empty and of equal size");
    }
    
    double sumPercentageError = 0.0;
    size_t validCount = 0;
    
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != 0.0) {  // Avoid division by zero
            double percentageError = std::abs((actual[i] - predicted[i]) / actual[i]) * 100.0;
            sumPercentageError += percentageError;
            validCount++;
        }
    }
    
    return validCount > 0 ? sumPercentageError / validCount : 0.0;
}

// Calculate R-squared
double Evaluator::calculateR2(const std::vector<double>& actual, const std::vector<double>& predicted) {
    if (actual.size() != predicted.size() || actual.empty()) {
        throw std::invalid_argument("Vectors must be non-empty and of equal size");
    }
    
    // Calculate mean of actual values
    double meanActual = std::accumulate(actual.begin(), actual.end(), 0.0) / actual.size();
    
    // Calculate sum of squares
    double totalSumSquares = 0.0;
    double residualSumSquares = 0.0;
    
    for (size_t i = 0; i < actual.size(); ++i) {
        totalSumSquares += (actual[i] - meanActual) * (actual[i] - meanActual);
        residualSumSquares += (actual[i] - predicted[i]) * (actual[i] - predicted[i]);
    }
    
    return totalSumSquares == 0.0 ? 1.0 : 1.0 - (residualSumSquares / totalSumSquares);
}

// Calculate residuals
std::vector<double> Evaluator::calculateResiduals(const std::vector<double>& actual, const std::vector<double>& predicted) {
    if (actual.size() != predicted.size()) {
        throw std::invalid_argument("Vectors must be of equal size");
    }
    
    std::vector<double> residuals;
    residuals.reserve(actual.size());
    
    for (size_t i = 0; i < actual.size(); ++i) {
        residuals.push_back(actual[i] - predicted[i]);
    }
    
    return residuals;
}

// Display results
void Evaluator::displayResults(const EvaluationResults& results) const {
    std::cout << "\n=== Evaluation Results ===" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "RMSE:  " << results.rmse << std::endl;
    std::cout << "MSE:   " << results.mse << std::endl;
    std::cout << "MAE:   " << results.mae << std::endl;
    std::cout << "R²:    " << results.rSquared << std::endl;
    std::cout << "MAPE:  " << results.meanAbsolutePercentageError << "%" << std::endl;
    std::cout << "Samples: " << results.predictions.size() << std::endl;
}

// Helper functions
double Evaluator::calculateMean(const std::vector<double>& values) const {
    if (values.empty()) return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double Evaluator::calculateVariance(const std::vector<double>& values) const {
    if (values.empty()) return 0.0;
    
    double mean = calculateMean(values);
    double variance = 0.0;
    
    for (double value : values) {
        variance += (value - mean) * (value - mean);
    }
    
    return variance / values.size();
}

double Evaluator::calculateStandardDeviation(const std::vector<double>& values) const {
    return std::sqrt(calculateVariance(values));
}
