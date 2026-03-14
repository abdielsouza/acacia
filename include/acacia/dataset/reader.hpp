#ifndef ACACIA_DATASET_READER_HPP_
#define ACACIA_DATASET_READER_HPP_

#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <variant>

namespace acacia::dataset
{
    // =============== SECTION 1 ===============
    // Specific dataset format data.
    // =========================================

    struct ExcelDataset
    {
        std::string file_path;
        std::string sheet_name;
        size_t sheet_index = 0;
    };

    struct CSVDataset
    {
        std::string file_path;
        char delimiter = ',';
        bool has_header = true;
    };

    struct ParquetDataset
    {
        std::string file_path;
    };

    struct LocalDBDataset
    {
        std::string connection_string;
        std::string table_name;
    };

    // =============== SECTION 2 ===============
    // Abstraction for using datasets.
    // =========================================

    // Type alias for data values (supports double, int, string, bool)
    using DataValue = std::variant<double, int64_t, std::string, bool>;

    template <typename T>
    class Dataset
    {
    public:
        Dataset() = default;
        virtual ~Dataset() = default;

        virtual size_t rows() const = 0;
        virtual size_t cols() const = 0;
    };

    template <>
    class Dataset<ExcelDataset>
    {
    private:
        ExcelDataset config_;
        std::vector<std::string> column_names_;
        std::vector<std::vector<DataValue>> data_;

    public:
        Dataset() = default;
        
        explicit Dataset(const ExcelDataset& config);

        size_t rows() const;
        size_t cols() const;

        const std::vector<std::string>& column_names() const;
        const std::vector<std::vector<DataValue>>& data() const;

        DataValue at(size_t row, size_t col) const;
        const std::vector<DataValue>& row(size_t row) const;
    };

    template <>
    class Dataset<CSVDataset>
    {
    private:
        CSVDataset config_;
        std::vector<std::string> column_names_;
        std::vector<std::vector<DataValue>> data_;

    public:
        Dataset() = default;
        
        explicit Dataset(const CSVDataset& config);

        size_t rows() const;
        size_t cols() const;

        const std::vector<std::string>& column_names() const;
        const std::vector<std::vector<DataValue>>& data() const;

        DataValue at(size_t row, size_t col) const;
        const std::vector<DataValue>& row(size_t row) const;
    };

    template <>
    class Dataset<ParquetDataset>
    {
    private:
        ParquetDataset config_;
        std::vector<std::string> column_names_;
        std::vector<std::vector<DataValue>> data_;

    public:
        Dataset() = default;
        
        explicit Dataset(const ParquetDataset& config);

        size_t rows() const;
        size_t cols() const;

        const std::vector<std::string>& column_names() const;
        const std::vector<std::vector<DataValue>>& data() const;

        DataValue at(size_t row, size_t col) const;
        const std::vector<DataValue>& row(size_t row) const;
    };

    template <>
    class Dataset<LocalDBDataset>
    {
    private:
        LocalDBDataset config_;
        std::vector<std::string> column_names_;
        std::vector<std::vector<DataValue>> data_;

    public:
        Dataset() = default;
        
        explicit Dataset(const LocalDBDataset& config);

        size_t rows() const;
        size_t cols() const;

        const std::vector<std::string>& column_names() const;
        const std::vector<std::vector<DataValue>>& data() const;

        DataValue at(size_t row, size_t col) const;
        const std::vector<DataValue>& row(size_t row) const;
    };

    // =============== SECTION 3 ===============
    // Functions to read files.
    // =========================================

    Dataset<ExcelDataset>   read_excel(const char* path, const char* sheet_name = nullptr);
    Dataset<CSVDataset>     read_csv(const char* path, char delimiter = ',', bool has_header = true);
    Dataset<ParquetDataset> read_parquet(const char* path);
    Dataset<LocalDBDataset> read_db(const char* connection_string, const char* table_name);
}

#endif