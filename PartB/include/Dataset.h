#ifndef DATASET_H
#define DATASET_H

#include "DataPoint.h"
#include <vector>
#include <string>
#include <random>

/**
 * @brief Dataset class for handling CPU performance data
 */
class Dataset {
private:
    std::vector<DataPoint> data;
    std::mt19937 rng;

public:
    // Constructor
    Dataset();
    
    // Destructor
    ~Dataset() = default;

    // Load data from file
    bool loadFromFile(const std::string& filename);
    
    // Get data
    const std::vector<DataPoint>& getData() const { return data; }
    std::vector<DataPoint>& getData() { return data; }
    
    // Size
    size_t size() const { return data.size(); }
    bool empty() const { return data.empty(); }
    
    // Access elements
    const DataPoint& operator[](size_t index) const;
    DataPoint& operator[](size_t index);
    
    // Add data point
    void addDataPoint(const DataPoint& point);
    
    // Clear data
    void clear();
    
    // Split dataset into training and testing sets
    void split(double trainRatio, Dataset& trainSet, Dataset& testSet);
    
    // Shuffle data
    void shuffle();
    
    // Get feature matrix (X) and target vector (y)
    void getMatrices(std::vector<std::vector<double>>& X, std::vector<double>& y) const;
    
    // Display statistics
    void displayStatistics() const;
    
    // Display first n data points
    void displaySample(size_t n = 5) const;

private:
    // Helper function to parse CSV line
    std::vector<std::string> parseLine(const std::string& line) const;
    
    // Helper function to trim whitespace
    std::string trim(const std::string& str) const;
};

#endif // DATASET_H
