#include "acacia/dataset/reader.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <iostream>

namespace acacia::dataset
{
    // =============== UTILITY FUNCTIONS ===============

    namespace detail
    {
        // Helper function to trim whitespace from strings
        std::string trim(const std::string& str)
        {
            size_t first = str.find_first_not_of(" \t\r\n");
            if (first == std::string::npos) return "";
            size_t last = str.find_last_not_of(" \t\r\n");
            return str.substr(first, (last - first + 1));
        }

        // Helper function to try converting string to appropriate data type
        DataValue try_convert_value(const std::string& str)
        {
            if (str.empty()) {
                return std::string("");
            }

            // Try bool
            std::string lower = str;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
            if (lower == "true" || lower == "yes" || lower == "1") {
                return true;
            }
            if (lower == "false" || lower == "no" || lower == "0") {
                return false;
            }

            // Try integer
            try {
                size_t idx;
                int64_t int_val = std::stoll(str, &idx);
                if (idx == str.length()) {
                    return int_val;
                }
            } catch (...) {
                // Not an integer
            }

            // Try double
            try {
                size_t idx;
                double double_val = std::stod(str, &idx);
                // Allow some tolerance for double conversion (e.g., "3.14" is valid)
                if (idx >= str.length() - 1) {
                    return double_val;
                }
            } catch (...) {
                // Not a double
            }

            // Default to string
            return str;
        }

        // Split a CSV line considering quoted fields
        std::vector<std::string> split_csv_line(const std::string& line, char delimiter)
        {
            std::vector<std::string> fields;
            std::string field;
            bool in_quotes = false;

            for (size_t i = 0; i < line.length(); ++i) {
                char c = line[i];

                if (c == '"') {
                    in_quotes = !in_quotes;
                } else if (c == delimiter && !in_quotes) {
                    fields.push_back(trim(field));
                    field.clear();
                } else {
                    field += c;
                }
            }
            fields.push_back(trim(field));
            return fields;
        }
    }

    // =============== DATASET<CSVDATASET> IMPLEMENTATION ===============

    Dataset<CSVDataset>::Dataset(const CSVDataset& config) : config_(config)
    {
        std::ifstream file(config.file_path);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open CSV file: " + config.file_path);
        }

        std::string line;
        bool first_line = true;

        while (std::getline(file, line)) {
            if (line.empty()) continue;

            auto fields = detail::split_csv_line(line, config.delimiter);

            if (first_line && config.has_header) {
                column_names_ = fields;
                first_line = false;
            } else {
                std::vector<DataValue> row;
                for (const auto& field : fields) {
                    row.push_back(detail::try_convert_value(field));
                }
                data_.push_back(row);

                if (first_line && !config.has_header) {
                    // Create default column names (Col_0, Col_1, ...)
                    for (size_t i = 0; i < fields.size(); ++i) {
                        column_names_.push_back("Col_" + std::to_string(i));
                    }
                    first_line = false;
                }
            }
        }

        file.close();
    }

    DataValue Dataset<CSVDataset>::at(size_t row, size_t col) const
    {
        if (row >= data_.size() || col >= data_[row].size()) {
            throw std::out_of_range("Index out of range");
        }
        return data_[row][col];
    }

    const std::vector<DataValue>& Dataset<CSVDataset>::row(size_t row) const
    {
        if (row >= data_.size()) {
            throw std::out_of_range("Row index out of range");
        }
        return data_[row];
    }

    size_t Dataset<CSVDataset>::rows() const
    {
        return data_.size();
    }

    size_t Dataset<CSVDataset>::cols() const
    {
        return column_names_.size();
    }

    const std::vector<std::string>& Dataset<CSVDataset>::column_names() const
    {
        return column_names_;
    }

    const std::vector<std::vector<DataValue>>& Dataset<CSVDataset>::data() const
    {
        return data_;
    }

    // =============== DATASET<EXCELDATASET> IMPLEMENTATION ===============

    Dataset<ExcelDataset>::Dataset(const ExcelDataset& config) : config_(config)
    {
        // Placeholder: Excel parsing would require external library
        // For now, we'll create a dummy dataset structure
        column_names_ = {"Excel Column 1", "Excel Column 2"};
        
        // Empty dataset - would be populated by excel parsing library
        std::cerr << "Warning: Excel file reading not yet fully implemented. "
                  << "Please use read_csv or read_parquet instead.\n";
    }

    DataValue Dataset<ExcelDataset>::at(size_t row, size_t col) const
    {
        if (row >= data_.size() || col >= data_[row].size()) {
            throw std::out_of_range("Index out of range");
        }
        return data_[row][col];
    }

    const std::vector<DataValue>& Dataset<ExcelDataset>::row(size_t row) const
    {
        if (row >= data_.size()) {
            throw std::out_of_range("Row index out of range");
        }
        return data_[row];
    }

    size_t Dataset<ExcelDataset>::rows() const
    {
        return data_.size();
    }

    size_t Dataset<ExcelDataset>::cols() const
    {
        return column_names_.size();
    }

    const std::vector<std::string>& Dataset<ExcelDataset>::column_names() const
    {
        return column_names_;
    }

    const std::vector<std::vector<DataValue>>& Dataset<ExcelDataset>::data() const
    {
        return data_;
    }

    // =============== DATASET<PARQUETDATASET> IMPLEMENTATION ===============

    Dataset<ParquetDataset>::Dataset(const ParquetDataset& config) : config_(config)
    {
        // Placeholder: Parquet parsing would require external library (arrow, parquet-cpp, etc.)
        column_names_ = {"Parquet Column 1", "Parquet Column 2"};
        
        // Empty dataset - would be populated by parquet parsing library
        std::cerr << "Warning: Parquet file reading not yet fully implemented. "
                  << "Please use read_csv instead.\n";
    }

    DataValue Dataset<ParquetDataset>::at(size_t row, size_t col) const
    {
        if (row >= data_.size() || col >= data_[row].size()) {
            throw std::out_of_range("Index out of range");
        }
        return data_[row][col];
    }

    const std::vector<DataValue>& Dataset<ParquetDataset>::row(size_t row) const
    {
        if (row >= data_.size()) {
            throw std::out_of_range("Row index out of range");
        }
        return data_[row];
    }

    size_t Dataset<ParquetDataset>::rows() const
    {
        return data_.size();
    }

    size_t Dataset<ParquetDataset>::cols() const
    {
        return column_names_.size();
    }

    const std::vector<std::string>& Dataset<ParquetDataset>::column_names() const
    {
        return column_names_;
    }

    const std::vector<std::vector<DataValue>>& Dataset<ParquetDataset>::data() const
    {
        return data_;
    }

    // =============== DATASET<LOCALDBDATASET> IMPLEMENTATION ===============

    Dataset<LocalDBDataset>::Dataset(const LocalDBDataset& config) : config_(config)
    {
        // Placeholder: Database connection would require external library (sqlite, postgres, etc.)
        column_names_ = {"DB Column 1", "DB Column 2"};
        
        // Empty dataset - would be populated by database query
        std::cerr << "Warning: Database reading not yet fully implemented. "
                  << "Please use read_csv instead.\n";
    }

    DataValue Dataset<LocalDBDataset>::at(size_t row, size_t col) const
    {
        if (row >= data_.size() || col >= data_[row].size()) {
            throw std::out_of_range("Index out of range");
        }
        return data_[row][col];
    }

    const std::vector<DataValue>& Dataset<LocalDBDataset>::row(size_t row) const
    {
        if (row >= data_.size()) {
            throw std::out_of_range("Row index out of range");
        }
        return data_[row];
    }

    size_t Dataset<LocalDBDataset>::rows() const
    {
        return data_.size();
    }

    size_t Dataset<LocalDBDataset>::cols() const
    {
        return column_names_.size();
    }

    const std::vector<std::string>& Dataset<LocalDBDataset>::column_names() const
    {
        return column_names_;
    }

    const std::vector<std::vector<DataValue>>& Dataset<LocalDBDataset>::data() const
    {
        return data_;
    }

    // =============== READ FUNCTIONS ===============

    Dataset<CSVDataset> read_csv(const char* path, char delimiter, bool has_header)
    {
        if (path == nullptr) {
            throw std::invalid_argument("Path cannot be null");
        }
        CSVDataset config{path, delimiter, has_header};
        return Dataset<CSVDataset>(config);
    }

    Dataset<ExcelDataset> read_excel(const char* path, const char* sheet_name)
    {
        if (path == nullptr) {
            throw std::invalid_argument("Path cannot be null");
        }
        ExcelDataset config{path, sheet_name ? std::string(sheet_name) : "", 0};
        return Dataset<ExcelDataset>(config);
    }

    Dataset<ParquetDataset> read_parquet(const char* path)
    {
        if (path == nullptr) {
            throw std::invalid_argument("Path cannot be null");
        }
        ParquetDataset config{path};
        return Dataset<ParquetDataset>(config);
    }

    Dataset<LocalDBDataset> read_db(const char* connection_string, const char* table_name)
    {
        if (connection_string == nullptr || table_name == nullptr) {
            throw std::invalid_argument("Arguments cannot be null");
        }
        LocalDBDataset config{connection_string, table_name};
        return Dataset<LocalDBDataset>(config);
    }
}
