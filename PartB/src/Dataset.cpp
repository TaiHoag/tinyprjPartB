#include "../include/Dataset.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <random>
#include <chrono>

// Constructor
Dataset::Dataset() : rng(std::chrono::steady_clock::now().time_since_epoch().count()) {}

// Load data from CSV file
bool Dataset::loadFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }
    
    data.clear();
    std::string line;
    int lineNumber = 0;
    
    while (std::getline(file, line)) {
        lineNumber++;
        line = trim(line);
        
        // Skip empty lines
        if (line.empty()) {
            continue;
        }
        
        // Parse CSV line
        std::vector<std::string> tokens = parseLine(line);
        
        // Validate number of columns
        if (tokens.size() != 10) {
            std::cerr << "Warning: Line " << lineNumber << " has " << tokens.size() 
                      << " columns instead of 10. Skipping." << std::endl;
            continue;
        }
        
        try {
            // Create DataPoint from parsed tokens
            DataPoint point;
            point.setVendor(trim(tokens[0]));
            point.setModel(trim(tokens[1]));
            point.setMYCT(std::stoi(tokens[2]));
            point.setMMIN(std::stoi(tokens[3]));
            point.setMMAX(std::stoi(tokens[4]));
            point.setCACH(std::stoi(tokens[5]));
            point.setCHMIN(std::stoi(tokens[6]));
            point.setCHMAX(std::stoi(tokens[7]));
            point.setPRP(std::stoi(tokens[8]));
            point.setERP(std::stoi(tokens[9]));
            
            data.push_back(point);
        }
        catch (const std::exception& e) {
            std::cerr << "Warning: Error parsing line " << lineNumber 
                      << ": " << e.what() << ". Skipping." << std::endl;
            continue;
        }
    }
    
    file.close();
    std::cout << "Successfully loaded " << data.size() << " data points from " << filename << std::endl;
    return !data.empty();
}

// Access operators
const DataPoint& Dataset::operator[](size_t index) const {
    if (index >= data.size()) {
        throw std::out_of_range("Dataset index out of range");
    }
    return data[index];
}

DataPoint& Dataset::operator[](size_t index) {
    if (index >= data.size()) {
        throw std::out_of_range("Dataset index out of range");
    }
    return data[index];
}

// Add data point
void Dataset::addDataPoint(const DataPoint& point) {
    data.push_back(point);
}

// Clear data
void Dataset::clear() {
    data.clear();
}

// Split dataset into training and testing sets
void Dataset::split(double trainRatio, Dataset& trainSet, Dataset& testSet) {
    if (trainRatio < 0.0 || trainRatio > 1.0) {
        throw std::invalid_argument("Train ratio must be between 0 and 1");
    }
    
    // Shuffle data first
    shuffle();
    
    size_t trainSize = static_cast<size_t>(data.size() * trainRatio);
    
    trainSet.clear();
    testSet.clear();
    
    // Add points to training set
    for (size_t i = 0; i < trainSize; ++i) {
        trainSet.addDataPoint(data[i]);
    }
    
    // Add remaining points to test set
    for (size_t i = trainSize; i < data.size(); ++i) {
        testSet.addDataPoint(data[i]);
    }
    
    std::cout << "Dataset split: " << trainSet.size() << " training samples, " 
              << testSet.size() << " test samples" << std::endl;
}

// Shuffle data
void Dataset::shuffle() {
    std::shuffle(data.begin(), data.end(), rng);
}

// Get feature matrix (X) and target vector (y)
void Dataset::getMatrices(std::vector<std::vector<double>>& X, std::vector<double>& y) const {
    X.clear();
    y.clear();
    
    X.reserve(data.size());
    y.reserve(data.size());
    
    for (const auto& point : data) {
        X.push_back(point.getFeatureVector());
        y.push_back(point.getTarget());
    }
}

// Display statistics
void Dataset::displayStatistics() const {
    if (data.empty()) {
        std::cout << "Dataset is empty." << std::endl;
        return;
    }
    
    std::cout << "\n=== Dataset Statistics ===" << std::endl;
    std::cout << "Number of samples: " << data.size() << std::endl;
    
    // Calculate statistics for each feature
    std::vector<std::string> featureNames = {"MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP"};
    
    for (size_t i = 0; i < featureNames.size(); ++i) {
        std::vector<double> values;
        values.reserve(data.size());
        
        for (const auto& point : data) {
            if (i < 6) {
                values.push_back(point.getFeatureVector()[i]);
            } else {
                values.push_back(point.getTarget());
            }
        }
        
        // Calculate min, max, mean, std
        double minVal = *std::min_element(values.begin(), values.end());
        double maxVal = *std::max_element(values.begin(), values.end());
        double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
        
        double variance = 0.0;
        for (double val : values) {
            variance += (val - mean) * (val - mean);
        }
        variance /= values.size();
        double stdDev = std::sqrt(variance);
        
        std::cout << std::setw(8) << featureNames[i] 
                  << ": Min=" << std::setw(8) << std::fixed << std::setprecision(2) << minVal
                  << ", Max=" << std::setw(8) << maxVal
                  << ", Mean=" << std::setw(8) << mean
                  << ", Std=" << std::setw(8) << stdDev << std::endl;
    }
}

// Display sample data
void Dataset::displaySample(size_t n) const {
    if (data.empty()) {
        std::cout << "Dataset is empty." << std::endl;
        return;
    }
    
    size_t samplesToShow = std::min(n, data.size());
    
    std::cout << "\n=== Sample Data (" << samplesToShow << " points) ===" << std::endl;
    std::cout << std::setw(12) << "Vendor" 
              << std::setw(15) << "Model"
              << std::setw(8) << "MYCT"
              << std::setw(8) << "MMIN"
              << std::setw(8) << "MMAX"
              << std::setw(8) << "CACH"
              << std::setw(8) << "CHMIN"
              << std::setw(8) << "CHMAX"
              << std::setw(8) << "PRP"
              << std::setw(8) << "ERP" << std::endl;
    
    std::cout << std::string(100, '-') << std::endl;
    
    for (size_t i = 0; i < samplesToShow; ++i) {
        data[i].display();
    }
}

// Helper function to parse CSV line
std::vector<std::string> Dataset::parseLine(const std::string& line) const {
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string token;
    
    while (std::getline(ss, token, ',')) {
        tokens.push_back(token);
    }
    
    return tokens;
}

// Helper function to trim whitespace
std::string Dataset::trim(const std::string& str) const {
    size_t start = str.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return "";
    }
    
    size_t end = str.find_last_not_of(" \t\r\n");
    return str.substr(start, end - start + 1);
}
