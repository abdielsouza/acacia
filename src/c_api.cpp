#include "acacia/c_api.h"

#include <atomic>
#include <mutex>
#include <string>
#include <vector>

#include "acacia/ml/regression.hpp"

namespace {

thread_local std::string g_last_error;

void set_last_error(const std::string& msg)
{
    g_last_error = msg;
}

} // namespace

extern "C" {

const char* acacia_get_last_error(void)
{
    return g_last_error.empty() ? "" : g_last_error.c_str();
}

struct acacia_lr {
    acacia::ml::LinearRegression<double> impl;
};

acacia_lr_t* acacia_lr_create(void)
{
    try {
        return new acacia_lr();
    } catch (const std::exception& ex) {
        set_last_error(ex.what());
        return nullptr;
    } catch (...) {
        set_last_error("Unknown allocation error");
        return nullptr;
    }
}

void acacia_lr_destroy(acacia_lr_t* model)
{
    delete model;
}

acacia_status_t acacia_lr_fit(
    acacia_lr_t* model,
    const double* X,
    size_t n_samples,
    size_t n_features,
    const double* y)
{
    if (!model || !X || !y) {
        set_last_error("Null pointer passed to acacia_lr_fit");
        return ACACIA_STATUS_ERROR;
    }

    try {
        std::vector<std::vector<double>> X_mat(n_samples, std::vector<double>(n_features));
        for (size_t i = 0; i < n_samples; ++i) {
            for (size_t j = 0; j < n_features; ++j) {
                X_mat[i][j] = X[i * n_features + j];
            }
        }

        std::vector<double> y_vec(n_samples);
        for (size_t i = 0; i < n_samples; ++i) {
            y_vec[i] = y[i];
        }

        model->impl.fit(X_mat, y_vec);
        return ACACIA_STATUS_OK;
    } catch (const std::exception& ex) {
        set_last_error(ex.what());
        return ACACIA_STATUS_ERROR;
    } catch (...) {
        set_last_error("Unknown error in acacia_lr_fit");
        return ACACIA_STATUS_ERROR;
    }
}

acacia_status_t acacia_lr_predict(
    const acacia_lr_t* model,
    const double* X,
    size_t n_samples,
    size_t n_features,
    double* out_predictions)
{
    if (!model || !X || !out_predictions) {
        set_last_error("Null pointer passed to acacia_lr_predict");
        return ACACIA_STATUS_ERROR;
    }

    try {
        std::vector<std::vector<double>> X_mat(n_samples, std::vector<double>(n_features));
        for (size_t i = 0; i < n_samples; ++i) {
            for (size_t j = 0; j < n_features; ++j) {
                X_mat[i][j] = X[i * n_features + j];
            }
        }

        auto results = model->impl.predict(X_mat);
        if (results.size() != n_samples) {
            set_last_error("Unexpected prediction size");
            return ACACIA_STATUS_ERROR;
        }

        for (size_t i = 0; i < n_samples; ++i) {
            out_predictions[i] = results[i];
        }

        return ACACIA_STATUS_OK;
    } catch (const std::exception& ex) {
        set_last_error(ex.what());
        return ACACIA_STATUS_ERROR;
    } catch (...) {
        set_last_error("Unknown error in acacia_lr_predict");
        return ACACIA_STATUS_ERROR;
    }
}

acacia_status_t acacia_lr_get_weights(
    const acacia_lr_t* model,
    double* out_weights,
    size_t n_features)
{
    if (!model || !out_weights) {
        set_last_error("Null pointer passed to acacia_lr_get_weights");
        return ACACIA_STATUS_ERROR;
    }

    try {
        auto weights = model->impl.get_weights();
        size_t expected = n_features + 1;
        if (weights.size() != expected) {
            set_last_error("Model weights size does not match expected feature count");
            return ACACIA_STATUS_ERROR;
        }

        for (size_t i = 0; i < expected; ++i) {
            out_weights[i] = weights[i];
        }

        return ACACIA_STATUS_OK;
    } catch (const std::exception& ex) {
        set_last_error(ex.what());
        return ACACIA_STATUS_ERROR;
    } catch (...) {
        set_last_error("Unknown error in acacia_lr_get_weights");
        return ACACIA_STATUS_ERROR;
    }
}

} // extern "C"
