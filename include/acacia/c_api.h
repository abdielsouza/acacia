#ifndef ACACIA_C_API_H_
#define ACACIA_C_API_H_

#pragma once

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include "acacia/macros.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file acacia/c_api.h
 * @brief C API for the Acacia library.
 *
 * This header provides a C-compatible interface for the primary features of
 * Acacia, including machine learning models, metrics, computer vision filters,
 * distributions, dataset I/O, and numerical utilities.
 */

/** Status codes returned by the C API. */
typedef enum acacia_status {
    ACACIA_STATUS_OK = 0,
    ACACIA_STATUS_ERROR = 1,
} acacia_status_t;

/**
 * @brief Returns a human-readable message for the last error that occurred in the current thread.
 *
 * Note: The returned string is owned by the library and must not be freed by the caller.
 */
ACACIA_API const char* acacia_get_last_error(void);

// ---------------------------------------------------------------------------
// Machine Learning - Regression
// ---------------------------------------------------------------------------

/** Opaque handle to a LinearRegression model. */
typedef struct acacia_lr acacia_lr_t;

ACACIA_API acacia_lr_t* acacia_lr_create(void);
ACACIA_API void acacia_lr_destroy(acacia_lr_t* model);
ACACIA_API acacia_status_t acacia_lr_fit(
    acacia_lr_t* model,
    const double* X,
    size_t n_samples,
    size_t n_features,
    const double* y);
ACACIA_API acacia_status_t acacia_lr_predict(
    const acacia_lr_t* model,
    const double* X,
    size_t n_samples,
    size_t n_features,
    double* out_predictions);
ACACIA_API acacia_status_t acacia_lr_get_weights(
    const acacia_lr_t* model,
    double* out_weights,
    size_t n_features);

// ---------------------------------------------------------------------------
// Machine Learning - Losses and Metrics
// ---------------------------------------------------------------------------

ACACIA_API acacia_status_t acacia_mse_loss(
    const double* predictions,
    const double* targets,
    size_t n,
    double* out_loss);

ACACIA_API acacia_status_t acacia_mse_gradient(
    const double* predictions,
    const double* targets,
    size_t n,
    double* out_gradient); // size: n

ACACIA_API acacia_status_t acacia_mae_loss(
    const double* predictions,
    const double* targets,
    size_t n,
    double* out_loss);

ACACIA_API acacia_status_t acacia_mae_gradient(
    const double* predictions,
    const double* targets,
    size_t n,
    double* out_gradient); // size: n

ACACIA_API acacia_status_t acacia_binary_cross_entropy_loss(
    const double* predictions,
    const double* targets,
    size_t n,
    double* out_loss);

ACACIA_API acacia_status_t acacia_binary_cross_entropy_gradient(
    const double* predictions,
    const double* targets,
    size_t n,
    double* out_gradient); // size: n

ACACIA_API acacia_status_t acacia_categorical_cross_entropy_loss(
    const double* predictions, // row-major: n_samples * n_classes
    const double* targets,     // row-major: n_samples * n_classes
    size_t n_samples,
    size_t n_classes,
    double* out_loss);

ACACIA_API acacia_status_t acacia_categorical_cross_entropy_gradient(
    const double* predictions, // row-major: n_samples * n_classes
    const double* targets,     // row-major: n_samples * n_classes
    size_t n_samples,
    size_t n_classes,
    double* out_gradient); // row-major: n_samples * n_classes

ACACIA_API acacia_status_t acacia_mse_metric(
    const double* predictions,
    const double* targets,
    size_t n,
    double* out_value);

ACACIA_API acacia_status_t acacia_rmse_metric(
    const double* predictions,
    const double* targets,
    size_t n,
    double* out_value);

ACACIA_API acacia_status_t acacia_mae_metric(
    const double* predictions,
    const double* targets,
    size_t n,
    double* out_value);

ACACIA_API acacia_status_t acacia_r_squared_metric(
    const double* predictions,
    const double* targets,
    size_t n,
    double* out_value);

ACACIA_API acacia_status_t acacia_accuracy_metric(
    const double* predictions,
    const double* targets,
    size_t n,
    double* out_value);

ACACIA_API acacia_status_t acacia_precision_metric(
    const int* predictions,
    const int* targets,
    size_t n,
    double* out_value);

ACACIA_API acacia_status_t acacia_recall_metric(
    const int* predictions,
    const int* targets,
    size_t n,
    double* out_value);

ACACIA_API acacia_status_t acacia_f1_score_metric(
    const int* predictions,
    const int* targets,
    size_t n,
    double* out_value);

// ---------------------------------------------------------------------------
// Machine Learning - Optimizers
// ---------------------------------------------------------------------------

/** Opaque handles for optimizers. */
typedef struct acacia_optimizer_sgd acacia_optimizer_sgd_t;
typedef struct acacia_optimizer_sgdm acacia_optimizer_sgdm_t;
typedef struct acacia_optimizer_adam acacia_optimizer_adam_t;
typedef struct acacia_optimizer_rmsprop acacia_optimizer_rmsprop_t;

ACACIA_API acacia_optimizer_sgd_t* acacia_optimizer_sgd_create(double learning_rate);
ACACIA_API void acacia_optimizer_sgd_destroy(acacia_optimizer_sgd_t* opt);
ACACIA_API acacia_status_t acacia_optimizer_sgd_step(
    acacia_optimizer_sgd_t* opt,
    double* params,
    const double* gradients,
    size_t n);
ACACIA_API acacia_status_t acacia_optimizer_sgd_reset(acacia_optimizer_sgd_t* opt);

ACACIA_API acacia_optimizer_sgdm_t* acacia_optimizer_sgdm_create(double learning_rate, double momentum);
ACACIA_API void acacia_optimizer_sgdm_destroy(acacia_optimizer_sgdm_t* opt);
ACACIA_API acacia_status_t acacia_optimizer_sgdm_step(
    acacia_optimizer_sgdm_t* opt,
    double* params,
    const double* gradients,
    size_t n);
ACACIA_API acacia_status_t acacia_optimizer_sgdm_reset(acacia_optimizer_sgdm_t* opt);

ACACIA_API acacia_optimizer_adam_t* acacia_optimizer_adam_create(double learning_rate,
                                                                 double beta1,
                                                                 double beta2,
                                                                 double epsilon);
ACACIA_API void acacia_optimizer_adam_destroy(acacia_optimizer_adam_t* opt);
ACACIA_API acacia_status_t acacia_optimizer_adam_step(
    acacia_optimizer_adam_t* opt,
    double* params,
    const double* gradients,
    size_t n);
ACACIA_API acacia_status_t acacia_optimizer_adam_reset(acacia_optimizer_adam_t* opt);

ACACIA_API acacia_optimizer_rmsprop_t* acacia_optimizer_rmsprop_create(double learning_rate,
                                                                       double rho,
                                                                       double epsilon);
ACACIA_API void acacia_optimizer_rmsprop_destroy(acacia_optimizer_rmsprop_t* opt);
ACACIA_API acacia_status_t acacia_optimizer_rmsprop_step(
    acacia_optimizer_rmsprop_t* opt,
    double* params,
    const double* gradients,
    size_t n);
ACACIA_API acacia_status_t acacia_optimizer_rmsprop_reset(acacia_optimizer_rmsprop_t* opt);

// ---------------------------------------------------------------------------
// Computer Vision
// ---------------------------------------------------------------------------

/** Opaque handle for a convolution kernel. */
typedef struct acacia_cv_kernel acacia_cv_kernel_t;

ACACIA_API acacia_cv_kernel_t* acacia_cv_kernel_create(size_t rows,
                                                       size_t cols,
                                                       const double* data);
ACACIA_API void acacia_cv_kernel_destroy(acacia_cv_kernel_t* kernel);

ACACIA_API acacia_cv_kernel_t* acacia_cv_kernel_gaussian(size_t size, double sigma);
ACACIA_API acacia_cv_kernel_t* acacia_cv_kernel_sobel_x(void);
ACACIA_API acacia_cv_kernel_t* acacia_cv_kernel_sobel_y(void);
ACACIA_API acacia_cv_kernel_t* acacia_cv_kernel_laplacian(void);
ACACIA_API acacia_cv_kernel_t* acacia_cv_kernel_box_blur(size_t size);
ACACIA_API acacia_cv_kernel_t* acacia_cv_kernel_sharpen(void);

ACACIA_API acacia_status_t acacia_cv_convolve(
    const double* image,
    size_t rows,
    size_t cols,
    const acacia_cv_kernel_t* kernel,
    double* out_image);

ACACIA_API acacia_status_t acacia_cv_gaussian_blur(
    const double* image,
    size_t rows,
    size_t cols,
    size_t kernel_size,
    double sigma,
    double* out_image);

ACACIA_API acacia_status_t acacia_cv_box_blur(
    const double* image,
    size_t rows,
    size_t cols,
    size_t kernel_size,
    double* out_image);

ACACIA_API acacia_status_t acacia_cv_sobel_edge_detection(
    const double* image,
    size_t rows,
    size_t cols,
    double* out_image);

ACACIA_API acacia_status_t acacia_cv_laplacian_edge_detection(
    const double* image,
    size_t rows,
    size_t cols,
    double* out_image);

ACACIA_API acacia_status_t acacia_cv_sharpen(
    const double* image,
    size_t rows,
    size_t cols,
    double* out_image);

ACACIA_API acacia_status_t acacia_cv_median_filter(
    const double* image,
    size_t rows,
    size_t cols,
    size_t kernel_size,
    double* out_image);

ACACIA_API acacia_status_t acacia_cv_bilateral_filter(
    const double* image,
    size_t rows,
    size_t cols,
    double spatial_sigma,
    double intensity_sigma,
    size_t kernel_size,
    double* out_image);

// ---------------------------------------------------------------------------
// Statistics / Distributions
// ---------------------------------------------------------------------------

/** Opaque handles for distributions. */
typedef struct acacia_dist_normal acacia_dist_normal_t;
typedef struct acacia_dist_uniform acacia_dist_uniform_t;
typedef struct acacia_dist_bernoulli acacia_dist_bernoulli_t;
typedef struct acacia_dist_categorical acacia_dist_categorical_t;
typedef struct acacia_dist_exponential acacia_dist_exponential_t;

ACACIA_API acacia_dist_normal_t* acacia_dist_normal_create(double mean, double stddev, unsigned int seed);
ACACIA_API void acacia_dist_normal_destroy(acacia_dist_normal_t* dist);
ACACIA_API acacia_status_t acacia_dist_normal_pdf(const acacia_dist_normal_t* dist, double x, double* out);
ACACIA_API acacia_status_t acacia_dist_normal_cdf(const acacia_dist_normal_t* dist, double x, double* out);
ACACIA_API acacia_status_t acacia_dist_normal_sample(const acacia_dist_normal_t* dist, double* out);
ACACIA_API acacia_status_t acacia_dist_normal_sample_n(const acacia_dist_normal_t* dist, size_t n, double* out);

ACACIA_API acacia_dist_uniform_t* acacia_dist_uniform_create(double a, double b, unsigned int seed);
ACACIA_API void acacia_dist_uniform_destroy(acacia_dist_uniform_t* dist);
ACACIA_API acacia_status_t acacia_dist_uniform_pdf(const acacia_dist_uniform_t* dist, double x, double* out);
ACACIA_API acacia_status_t acacia_dist_uniform_cdf(const acacia_dist_uniform_t* dist, double x, double* out);
ACACIA_API acacia_status_t acacia_dist_uniform_sample(const acacia_dist_uniform_t* dist, double* out);
ACACIA_API acacia_status_t acacia_dist_uniform_sample_n(const acacia_dist_uniform_t* dist, size_t n, double* out);

ACACIA_API acacia_dist_bernoulli_t* acacia_dist_bernoulli_create(double p, unsigned int seed);
ACACIA_API void acacia_dist_bernoulli_destroy(acacia_dist_bernoulli_t* dist);
ACACIA_API acacia_status_t acacia_dist_bernoulli_pdf(const acacia_dist_bernoulli_t* dist, double x, double* out);
ACACIA_API acacia_status_t acacia_dist_bernoulli_cdf(const acacia_dist_bernoulli_t* dist, double x, double* out);
ACACIA_API acacia_status_t acacia_dist_bernoulli_sample(const acacia_dist_bernoulli_t* dist, double* out);
ACACIA_API acacia_status_t acacia_dist_bernoulli_sample_n(const acacia_dist_bernoulli_t* dist, size_t n, double* out);

ACACIA_API acacia_dist_categorical_t* acacia_dist_categorical_create(const double* probabilities, size_t n, unsigned int seed);
ACACIA_API void acacia_dist_categorical_destroy(acacia_dist_categorical_t* dist);
ACACIA_API acacia_status_t acacia_dist_categorical_sample(const acacia_dist_categorical_t* dist, int* out);
ACACIA_API acacia_status_t acacia_dist_categorical_sample_n(const acacia_dist_categorical_t* dist, size_t n, int* out);

ACACIA_API acacia_dist_exponential_t* acacia_dist_exponential_create(double lambda, unsigned int seed);
ACACIA_API void acacia_dist_exponential_destroy(acacia_dist_exponential_t* dist);
ACACIA_API acacia_status_t acacia_dist_exponential_pdf(const acacia_dist_exponential_t* dist, double x, double* out);
ACACIA_API acacia_status_t acacia_dist_exponential_cdf(const acacia_dist_exponential_t* dist, double x, double* out);
ACACIA_API acacia_status_t acacia_dist_exponential_sample(const acacia_dist_exponential_t* dist, double* out);
ACACIA_API acacia_status_t acacia_dist_exponential_sample_n(const acacia_dist_exponential_t* dist, size_t n, double* out);

// ---------------------------------------------------------------------------
// Dataset (CSV only)
// ---------------------------------------------------------------------------

/** Opaque handle for a CSV dataset reader. */
typedef struct acacia_dataset acacia_dataset_t;

ACACIA_API acacia_dataset_t* acacia_dataset_read_csv(const char* path, char delimiter, bool has_header);
ACACIA_API void acacia_dataset_destroy(acacia_dataset_t* dataset);
ACACIA_API acacia_status_t acacia_dataset_rows(const acacia_dataset_t* dataset, size_t* out_rows);
ACACIA_API acacia_status_t acacia_dataset_cols(const acacia_dataset_t* dataset, size_t* out_cols);
ACACIA_API acacia_status_t acacia_dataset_get_column_name(const acacia_dataset_t* dataset, size_t col, const char** out_name);
ACACIA_API acacia_status_t acacia_dataset_get_value_as_string(
    const acacia_dataset_t* dataset,
    size_t row,
    size_t col,
    const char** out_value);

ACACIA_API acacia_status_t acacia_write_csv(
    const char* path,
    const char** column_names,
    size_t n_cols,
    const double* data, // row-major matrix
    size_t n_rows,
    bool write_header,
    bool overwrite);

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/** Function signature for unary functions used by numerical utilities. */
typedef double (*acacia_utils_unary_function)(double x, void* user_data);

ACACIA_API acacia_status_t acacia_utils_derivative(
    acacia_utils_unary_function f,
    void* user_data,
    double x,
    double h,
    double* out);

ACACIA_API acacia_status_t acacia_utils_integral(
    acacia_utils_unary_function f,
    void* user_data,
    double a,
    double b,
    size_t n,
    double* out);

ACACIA_API acacia_status_t acacia_utils_find_root(
    acacia_utils_unary_function f,
    void* user_data,
    double a,
    double b,
    double tol,
    double* out);

#ifdef __cplusplus
}
#endif

#endif // ACACIA_C_API_H_
