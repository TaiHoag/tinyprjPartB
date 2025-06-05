#include "../include/DataPoint.h"
#include <iostream>
#include <iomanip>

// Default constructor
DataPoint::DataPoint() 
    : vendor(""), model(""), myct(0), mmin(0), mmax(0), 
      cach(0), chmin(0), chmax(0), prp(0), erp(0) {}

// Parameterized constructor
DataPoint::DataPoint(const std::string& vendor, const std::string& model,
                     int myct, int mmin, int mmax, int cach, 
                     int chmin, int chmax, int prp, int erp)
    : vendor(vendor), model(model), myct(myct), mmin(mmin), mmax(mmax),
      cach(cach), chmin(chmin), chmax(chmax), prp(prp), erp(erp) {}

// Get feature vector for regression (MYCT, MMIN, MMAX, CACH, CHMIN, CHMAX)
std::vector<double> DataPoint::getFeatureVector() const {
    return {static_cast<double>(myct), 
            static_cast<double>(mmin), 
            static_cast<double>(mmax),
            static_cast<double>(cach), 
            static_cast<double>(chmin), 
            static_cast<double>(chmax)};
}

// Display data point information
void DataPoint::display() const {
    std::cout << std::setw(12) << vendor 
              << std::setw(15) << model
              << std::setw(8) << myct
              << std::setw(8) << mmin
              << std::setw(8) << mmax
              << std::setw(8) << cach
              << std::setw(8) << chmin
              << std::setw(8) << chmax
              << std::setw(8) << prp
              << std::setw(8) << erp << std::endl;
}
