#ifndef ACACIA_DATASET_WRITER_HPP_
#define ACACIA_DATASET_WRITER_HPP_

#pragma once

#include <vector>
#include <string>
#include <variant>
#include <stdexcept>

namespace acacia::dataset
{
    // =============== SECTION 1 ===============
    // Specific dataset format configuration.
    // =========================================

    struct ExcelWriterConfig
    {
        std::string file_path;
        std::string sheet_name = "Sheet1";
        bool overwrite = true;
    };

    struct CSVWriterConfig
    {
        std::string file_path;
        char delimiter = ',';
        bool write_header = true;
        bool overwrite = true;
    };

    struct ParquetWriterConfig
    {
        std::string file_path;
        bool overwrite = true;
        int compression_level = 6;  // 0-9, where 9 is best compression
    };

    struct LocalDBWriterConfig
    {
        std::string connection_string;
        std::string table_name;
        bool create_table = true;
        bool overwrite = false;
    };

    // =============== SECTION 2 ===============
    // Type alias for data values.
    // =========================================

    using DataValue = std::variant<double, int64_t, std::string, bool>;

    // =============== SECTION 3 ===============
    // Writer base class and specializations.
    // =========================================

    template <typename T>
    class DatasetWriter
    {
    public:
        DatasetWriter() = default;
        virtual ~DatasetWriter() = default;

        virtual void write(const std::vector<std::string>& column_names,
                          const std::vector<std::vector<DataValue>>& data) = 0;
    };

    template <>
    class DatasetWriter<ExcelWriterConfig>
    {
    private:
        ExcelWriterConfig config_;

    public:
        explicit DatasetWriter(const ExcelWriterConfig& config);

        void write(const std::vector<std::string>& column_names,
                   const std::vector<std::vector<DataValue>>& data);
    };

    template <>
    class DatasetWriter<CSVWriterConfig>
    {
    private:
        CSVWriterConfig config_;

    public:
        explicit DatasetWriter(const CSVWriterConfig& config);

        void write(const std::vector<std::string>& column_names,
                   const std::vector<std::vector<DataValue>>& data);
    };

    template <>
    class DatasetWriter<ParquetWriterConfig>
    {
    private:
        ParquetWriterConfig config_;

    public:
        explicit DatasetWriter(const ParquetWriterConfig& config);

        void write(const std::vector<std::string>& column_names,
                   const std::vector<std::vector<DataValue>>& data);
    };

    template <>
    class DatasetWriter<LocalDBWriterConfig>
    {
    private:
        LocalDBWriterConfig config_;

    public:
        explicit DatasetWriter(const LocalDBWriterConfig& config);

        void write(const std::vector<std::string>& column_names,
                   const std::vector<std::vector<DataValue>>& data);
    };

    // =============== SECTION 4 ===============
    // Writer factory functions.
    // =========================================

    DatasetWriter<ExcelWriterConfig> create_excel_writer(const char* path,
                                                         const char* sheet_name = "Sheet1",
                                                         bool overwrite = true);

    DatasetWriter<CSVWriterConfig> create_csv_writer(const char* path,
                                                      char delimiter = ',',
                                                      bool write_header = true,
                                                      bool overwrite = true);

    DatasetWriter<ParquetWriterConfig> create_parquet_writer(const char* path,
                                                              bool overwrite = true,
                                                              int compression_level = 6);

    DatasetWriter<LocalDBWriterConfig> create_db_writer(const char* connection_string,
                                                         const char* table_name,
                                                         bool create_table = true,
                                                         bool overwrite = false);
}

#endif