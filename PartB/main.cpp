#include "include/Dataset.h"
#include "include/LinearRegression.h"
#include "include/Evaluator.h"
#include <iostream>
#include <iomanip>
#include <chrono>

/**
 * @brief Main application for CPU Performance Linear Regression Prediction
 * 
 * This program implements a linear regression model to predict CPU relative performance
 * based on hardware specifications using the Computer Hardware dataset from UCI.
 * 
 * The model equation is:
 * PRP = x1*MYCT + x2*MMIN + x3*MMAX + x4*CACH + x5*CHMIN + x6*CHMAX
 * 
 * Where:
 * - MYCT: machine cycle time in nanoseconds
 * - MMIN: minimum main memory in kilobytes  
 * - MMAX: maximum main memory in kilobytes
 * - CACH: cache memory in kilobytes
 * - CHMIN: minimum channels in units
 * - CHMAX: maximum channels in units
 * - PRP: published relative performance (target variable)
 */

void printHeader() {
    std::cout << "=========================================================\n";
    std::cout << "      CPU PERFORMANCE LINEAR REGRESSION PREDICTOR\n";
    std::cout << "=========================================================\n";
    std::cout << "Dataset: Computer Hardware (UCI Machine Learning Repository)\n";
    std::cout << "Model: Linear Regression (Normal Equation)\n";
    std::cout << "Target: Published Relative Performance (PRP)\n";
    std::cout << "Features: MYCT, MMIN, MMAX, CACH, CHMIN, CHMAX\n";
    std::cout << "=========================================================\n\n";
}

void displayMenu() {
    std::cout << "\n=== OPTIONS ===" << std::endl;
    std::cout << "1. Load and display dataset statistics" << std::endl;
    std::cout << "2. Train linear regression model" << std::endl;
    std::cout << "3. Train with Ridge regularization" << std::endl;
    std::cout << "4. Evaluate model on test set" << std::endl;
    std::cout << "5. Make individual prediction" << std::endl;
    std::cout << "6. Perform cross-validation" << std::endl;
    std::cout << "7. Generate detailed evaluation report" << std::endl;
    std::cout << "8. Display model equation" << std::endl;
    std::cout << "9. Residual analysis" << std::endl;
    std::cout << "0. Exit" << std::endl;
    std::cout << "Choose an option: ";
}

void makeIndividualPrediction(const LinearRegression& model) {
    if (!model.getIsTrained()) {
        std::cout << "Error: Model has not been trained yet!" << std::endl;
        return;
    }
    
    std::cout << "\n=== Individual Prediction ===" << std::endl;
    std::cout << "Enter hardware specifications:" << std::endl;
    
    std::vector<double> features(6);
    std::vector<std::string> featureNames = {"MYCT (cycle time)", "MMIN (min memory)", 
                                           "MMAX (max memory)", "CACH (cache)", 
                                           "CHMIN (min channels)", "CHMAX (max channels)"};
    
    for (size_t i = 0; i < 6; ++i) {
        std::cout << featureNames[i] << ": ";
        std::cin >> features[i];
    }
    
    try {
        double prediction = model.predict(features);
        std::cout << "\nPredicted Relative Performance: " << std::fixed << std::setprecision(2) 
                  << prediction << std::endl;
    }
    catch (const std::exception& e) {
        std::cout << "Error making prediction: " << e.what() << std::endl;
    }
}

int main() {
    printHeader();
    
    // Initialize components
    Dataset fullDataset, trainDataset, testDataset;
    LinearRegression model;
    
    std::string dataFilePath = "Data/machine.data";
    bool dataLoaded = false;
    bool modelTrained = false;
    
    int choice;
    
    while (true) {
        displayMenu();
        std::cin >> choice;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        switch (choice) {
            case 1: {
                // Load and display dataset statistics
                std::cout << "\nLoading dataset from: " << dataFilePath << std::endl;
                
                if (fullDataset.loadFromFile(dataFilePath)) {
                    dataLoaded = true;
                    fullDataset.displayStatistics();
                    fullDataset.displaySample(10);
                    
                    // Split into train/test sets (80/20)
                    std::cout << "\nSplitting dataset (80% train, 20% test)..." << std::endl;
                    fullDataset.split(0.8, trainDataset, testDataset);
                } else {
                    std::cout << "Failed to load dataset!" << std::endl;
                }
                break;
            }
            
            case 2: {
                // Train linear regression model
                if (!dataLoaded) {
                    std::cout << "Please load the dataset first (option 1)!" << std::endl;
                    break;
                }
                
                std::cout << "\nTraining linear regression model..." << std::endl;
                if (model.train(trainDataset)) {
                    modelTrained = true;
                    model.displayModel();
                    model.displayEquation();
                } else {
                    std::cout << "Model training failed!" << std::endl;
                }
                break;
            }
            
            case 3: {
                // Train with Ridge regularization
                if (!dataLoaded) {
                    std::cout << "Please load the dataset first (option 1)!" << std::endl;
                    break;
                }
                
                double lambda;
                std::cout << "Enter regularization parameter (lambda, e.g., 0.01): ";
                std::cin >> lambda;
                
                std::cout << "\nTraining Ridge regression model..." << std::endl;
                if (model.trainWithRegularization(trainDataset, lambda)) {
                    modelTrained = true;
                    model.displayModel();
                    model.displayEquation();
                } else {
                    std::cout << "Ridge regression training failed!" << std::endl;
                }
                break;
            }
            
            case 4: {
                // Evaluate model on test set
                if (!modelTrained) {
                    std::cout << "Please train the model first (option 2 or 3)!" << std::endl;
                    break;
                }
                
                std::cout << "\nEvaluating model on test set..." << std::endl;
                
                Evaluator evaluator(&model);
                auto results = evaluator.evaluate(testDataset);
                evaluator.displayResults(results);
                evaluator.predictionComparison(testDataset, 15);
                break;
            }
            
            case 5: {
                // Make individual prediction
                makeIndividualPrediction(model);
                break;
            }
            
            case 6: {
                // Perform cross-validation
                if (!dataLoaded) {
                    std::cout << "Please load the dataset first (option 1)!" << std::endl;
                    break;
                }
                
                int folds;
                std::cout << "Enter number of folds (e.g., 5): ";
                std::cin >> folds;
                
                std::cout << "\nPerforming " << folds << "-fold cross-validation..." << std::endl;
                double avgRMSE = model.crossValidate(fullDataset, folds);
                
                if (avgRMSE >= 0) {
                    std::cout << "Cross-validation completed successfully!" << std::endl;
                } else {
                    std::cout << "Cross-validation failed!" << std::endl;
                }
                break;
            }
            
            case 7: {
                // Generate detailed evaluation report
                if (!modelTrained) {
                    std::cout << "Please train the model first (option 2 or 3)!" << std::endl;
                    break;
                }
                
                std::cout << "\nGenerating detailed evaluation report..." << std::endl;
                
                Evaluator evaluator(&model);
                evaluator.generateReport(testDataset, "evaluation_report.txt");
                evaluator.residualAnalysis(testDataset);
                break;
            }
            
            case 8: {
                // Display model equation
                if (!modelTrained) {
                    std::cout << "Please train the model first (option 2 or 3)!" << std::endl;
                    break;
                }
                
                model.displayModel();
                model.displayEquation();
                break;
            }
            
            case 9: {
                // Residual analysis
                if (!modelTrained) {
                    std::cout << "Please train the model first (option 2 or 3)!" << std::endl;
                    break;
                }
                
                Evaluator evaluator(&model);
                evaluator.residualAnalysis(testDataset);
                break;
            }
            
            case 0: {
                std::cout << "\nThank you for using CPU Performance Predictor!" << std::endl;
                return 0;
            }
            
            default: {
                std::cout << "Invalid option! Please choose 0-9." << std::endl;
                break;
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "\nOperation completed in " << duration.count() << " ms" << std::endl;
        
        std::cout << "\nPress Enter to continue...";
        std::cin.ignore();
        std::cin.get();
    }
    
    return 0;
}
