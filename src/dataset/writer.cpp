#include "acacia/dataset/writer.hpp"
#include <fstream>
#include <sstream>
#include <filesystem>
#include <iostream>
#include <iomanip>

namespace acacia::dataset
{
    // =============== UTILITY FUNCTIONS ===============

    namespace detail
    {
        // Helper function to convert DataValue to string
        std::string value_to_string(const DataValue& value)
        {
            if (std::holds_alternative<double>(value)) {
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(6) << std::get<double>(value);
                return oss.str();
            } else if (std::holds_alternative<int64_t>(value)) {
                return std::to_string(std::get<int64_t>(value));
            } else if (std::holds_alternative<std::string>(value)) {
                return std::get<std::string>(value);
            } else if (std::holds_alternative<bool>(value)) {
                return std::get<bool>(value) ? "true" : "false";
            }
            return "";
        }

        // Escape CSV fields that contain delimiters, quotes, or newlines
        std::string escape_csv_field(const std::string& field, char delimiter)
        {
            bool needs_quotes = false;

            // Check if field needs escaping
            if (field.find(delimiter) != std::string::npos ||
                field.find('"') != std::string::npos ||
                field.find('\n') != std::string::npos ||
                field.find('\r') != std::string::npos) {
                needs_quotes = true;
            }

            if (!needs_quotes) {
                return field;
            }

            // Escape internal quotes by doubling them
            std::string escaped;
            escaped += '"';
            for (char c : field) {
                if (c == '"') {
                    escaped += "\"\"";
                } else {
                    escaped += c;
                }
            }
            escaped += '"';
            return escaped;
        }

        // Write CSV line
        void write_csv_line(std::ofstream& file,
                           const std::vector<std::string>& fields,
                           char delimiter)
        {
            for (size_t i = 0; i < fields.size(); ++i) {
                if (i > 0) {
                    file << delimiter;
                }
                file << escape_csv_field(fields[i], delimiter);
            }
            file << "\n";
        }

        // Check if file exists
        bool file_exists(const std::string& path)
        {
            return std::filesystem::exists(path);
        }

        // Backup existing file
        void backup_file(const std::string& path)
        {
            if (file_exists(path)) {
                std::string backup_path = path + ".bak";
                std::filesystem::copy_file(path, backup_path,
                                          std::filesystem::copy_options::overwrite_existing);
            }
        }
    }

    // =============== DATASETWRITER<CSVWRITERCONFIG> IMPLEMENTATION ===============

    DatasetWriter<CSVWriterConfig>::DatasetWriter(const CSVWriterConfig& config) : config_(config) {}

    void DatasetWriter<CSVWriterConfig>::write(
        const std::vector<std::string>& column_names,
        const std::vector<std::vector<DataValue>>& data)
    {
        if (column_names.empty() || data.empty()) {
            throw std::invalid_argument("Column names and data cannot be empty");
        }

        // Check file overwrite
        if (!config_.overwrite && detail::file_exists(config_.file_path)) {
            throw std::runtime_error("File already exists: " + config_.file_path);
        }

        std::ofstream file(config_.file_path);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open CSV file for writing: " + config_.file_path);
        }

        // Write header
        if (config_.write_header) {
            detail::write_csv_line(file, column_names, config_.delimiter);
        }

        // Write data rows
        for (const auto& row : data) {
            std::vector<std::string> string_row;
            for (const auto& value : row) {
                string_row.push_back(detail::value_to_string(value));
            }
            detail::write_csv_line(file, string_row, config_.delimiter);
        }

        file.close();
    }

    // =============== DATASETWRITER<EXCELDATASET> IMPLEMENTATION ===============

    DatasetWriter<ExcelWriterConfig>::DatasetWriter(const ExcelWriterConfig& config) : config_(config) {}

    void DatasetWriter<ExcelWriterConfig>::write(
        const std::vector<std::string>& column_names,
        const std::vector<std::vector<DataValue>>& data)
    {
        if (column_names.empty() || data.empty()) {
            throw std::invalid_argument("Column names and data cannot be empty");
        }

        // Placeholder: Excel writing would require external library
        std::cerr << "Warning: Excel file writing not yet fully implemented. "
                  << "File path: " << config_.file_path << "\n"
                  << "Sheet name: " << config_.sheet_name << "\n"
                  << "Rows: " << data.size() << ", Columns: " << column_names.size() << "\n";
    }

    // =============== DATASETWRITER<PARQUETDATASET> IMPLEMENTATION ===============

    DatasetWriter<ParquetWriterConfig>::DatasetWriter(const ParquetWriterConfig& config) : config_(config) {}

    void DatasetWriter<ParquetWriterConfig>::write(
        const std::vector<std::string>& column_names,
        const std::vector<std::vector<DataValue>>& data)
    {
        if (column_names.empty() || data.empty()) {
            throw std::invalid_argument("Column names and data cannot be empty");
        }

        // Placeholder: Parquet writing would require external library (arrow, parquet-cpp, etc.)
        std::cerr << "Warning: Parquet file writing not yet fully implemented. "
                  << "File path: " << config_.file_path << "\n"
                  << "Compression level: " << config_.compression_level << "\n"
                  << "Rows: " << data.size() << ", Columns: " << column_names.size() << "\n";
    }

    // =============== DATASETWRITER<LOCALDBWRITERCONFIG> IMPLEMENTATION ===============

    DatasetWriter<LocalDBWriterConfig>::DatasetWriter(const LocalDBWriterConfig& config) : config_(config) {}

    void DatasetWriter<LocalDBWriterConfig>::write(
        const std::vector<std::string>& column_names,
        const std::vector<std::vector<DataValue>>& data)
    {
        if (column_names.empty() || data.empty()) {
            throw std::invalid_argument("Column names and data cannot be empty");
        }

        // Placeholder: Database writing would require external library (sqlite, postgres, etc.)
        std::cerr << "Warning: Database writing not yet fully implemented. "
                  << "Connection: " << config_.connection_string << "\n"
                  << "Table: " << config_.table_name << "\n"
                  << "Create table: " << (config_.create_table ? "yes" : "no") << "\n"
                  << "Rows: " << data.size() << ", Columns: " << column_names.size() << "\n";
    }

    // =============== WRITER FACTORY FUNCTIONS ===============

    DatasetWriter<ExcelWriterConfig> create_excel_writer(const char* path,
                                                         const char* sheet_name,
                                                         bool overwrite)
    {
        if (path == nullptr) {
            throw std::invalid_argument("Path cannot be null");
        }
        ExcelWriterConfig config{path, sheet_name ? std::string(sheet_name) : "Sheet1", overwrite};
        return DatasetWriter<ExcelWriterConfig>(config);
    }

    DatasetWriter<CSVWriterConfig> create_csv_writer(const char* path,
                                                      char delimiter,
                                                      bool write_header,
                                                      bool overwrite)
    {
        if (path == nullptr) {
            throw std::invalid_argument("Path cannot be null");
        }
        CSVWriterConfig config{path, delimiter, write_header, overwrite};
        return DatasetWriter<CSVWriterConfig>(config);
    }

    DatasetWriter<ParquetWriterConfig> create_parquet_writer(const char* path,
                                                              bool overwrite,
                                                              int compression_level)
    {
        if (path == nullptr) {
            throw std::invalid_argument("Path cannot be null");
        }
        if (compression_level < 0 || compression_level > 9) {
            throw std::invalid_argument("Compression level must be between 0 and 9");
        }
        ParquetWriterConfig config{path, overwrite, compression_level};
        return DatasetWriter<ParquetWriterConfig>(config);
    }

    DatasetWriter<LocalDBWriterConfig> create_db_writer(const char* connection_string,
                                                         const char* table_name,
                                                         bool create_table,
                                                         bool overwrite)
    {
        if (connection_string == nullptr || table_name == nullptr) {
            throw std::invalid_argument("Arguments cannot be null");
        }
        LocalDBWriterConfig config{connection_string, table_name, create_table, overwrite};
        return DatasetWriter<LocalDBWriterConfig>(config);
    }
}
