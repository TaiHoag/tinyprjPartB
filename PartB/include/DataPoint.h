#ifndef DATAPOINT_H
#define DATAPOINT_H

#include <string>
#include <vector>

/**
 * @brief Represents a single data point from the CPU performance dataset
 */
class DataPoint {
private:
    std::string vendor;
    std::string model;
    int myct;       // machine cycle time in nanoseconds
    int mmin;       // minimum main memory in kilobytes
    int mmax;       // maximum main memory in kilobytes
    int cach;       // cache memory in kilobytes
    int chmin;      // minimum channels in units
    int chmax;      // maximum channels in units
    int prp;        // published relative performance (target)
    int erp;        // estimated relative performance

public:
    // Constructor
    DataPoint();
    DataPoint(const std::string& vendor, const std::string& model,
              int myct, int mmin, int mmax, int cach, 
              int chmin, int chmax, int prp, int erp);

    // Getters
    std::string getVendor() const { return vendor; }
    std::string getModel() const { return model; }
    int getMYCT() const { return myct; }
    int getMMIN() const { return mmin; }
    int getMMAX() const { return mmax; }
    int getCACH() const { return cach; }
    int getCHMIN() const { return chmin; }
    int getCHMAX() const { return chmax; }
    int getPRP() const { return prp; }
    int getERP() const { return erp; }

    // Setters
    void setVendor(const std::string& v) { vendor = v; }
    void setModel(const std::string& m) { model = m; }
    void setMYCT(int m) { myct = m; }
    void setMMIN(int m) { mmin = m; }
    void setMMAX(int m) { mmax = m; }
    void setCACH(int c) { cach = c; }
    void setCHMIN(int c) { chmin = c; }
    void setCHMAX(int c) { chmax = c; }
    void setPRP(int p) { prp = p; }
    void setERP(int e) { erp = e; }

    // Get feature vector for regression (excluding vendor and model)
    std::vector<double> getFeatureVector() const;
    
    // Get target value
    double getTarget() const { return static_cast<double>(prp); }

    // Display
    void display() const;
};

#endif // DATAPOINT_H
