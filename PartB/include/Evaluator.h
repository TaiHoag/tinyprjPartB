#ifndef EVALUATOR_H
#define EVALUATOR_H

#include "LinearRegression.h"
#include "Dataset.h"
#include <vector>
#include <string>

/**
 * @brief Evaluator class for comprehensive model evaluation
 */
class Evaluator {
private:
    LinearRegression* model;
    
public:
    // Constructor
    Evaluator(LinearRegression* model);
    
    // Destructor
    ~Evaluator() = default;

    // Comprehensive evaluation
    struct EvaluationResults {
        double rmse;
        double mse;
        double mae;
        double rSquared;
        double meanAbsolutePercentageError;
        std::vector<double> predictions;
        std::vector<double> actuals;
        std::vector<double> residuals;
    };
    
    // Evaluate model on test set
    EvaluationResults evaluate(const Dataset& testData) const;
    
    // Generate detailed evaluation report
    void generateReport(const Dataset& testData, const std::string& filename = "") const;
    
    // Residual analysis
    void residualAnalysis(const Dataset& testData) const;
    
    // Prediction vs Actual comparison
    void predictionComparison(const Dataset& testData, size_t numSamples = 10) const;
    
    // Calculate various metrics
    static double calculateMAPE(const std::vector<double>& actual, const std::vector<double>& predicted);
    static double calculateR2(const std::vector<double>& actual, const std::vector<double>& predicted);
    static std::vector<double> calculateResiduals(const std::vector<double>& actual, const std::vector<double>& predicted);
    
    // Display results in formatted way
    void displayResults(const EvaluationResults& results) const;

private:
    // Helper functions
    double calculateMean(const std::vector<double>& values) const;
    double calculateVariance(const std::vector<double>& values) const;
    double calculateStandardDeviation(const std::vector<double>& values) const;
};

#endif // EVALUATOR_H
