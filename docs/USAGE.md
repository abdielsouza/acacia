# Uso da Biblioteca Acacia

Este documento mostra como utilizar os principais módulos da biblioteca **Acacia** em seus projetos C++.

---

## Estrutura de nomes (namespaces)

A Acacia está organizada em namespaces que refletem seus módulos:

- `acacia::ml` — Aprendizado de máquina (regressão, otimização, métricas)
- `acacia::cv` — Visão computacional (filtros, kernels)
- `acacia::stats` — Estatística (distribuições, amostragem, PDF/CDF)
- `acacia::dataset` — Leitura e escrita de datasets (CSV, Excel, Parquet, DB)
- `acacia::utils` — Utilitários (álgebra linear, cálculo numérico)

Os headers estão disponíveis em `include/acacia/`.

---

## 1) Machine Learning (ml)

### Exemplo: Regressão Linear

```cpp
#include "acacia/ml/regression.hpp"
#include "acacia/ml/metrics.hpp"

int main() {
    std::vector<std::vector<double>> X = {{1}, {2}, {3}, {4}, {5}};
    std::vector<double> y = {2, 4, 6, 8, 10};

    acacia::ml::LinearRegression<double> model;
    model.fit(X, y);

    auto predictions = model.predict(X);

    double mse = acacia::ml::MSE::evaluate(predictions, y);
    return 0;
}
```

### Recursos disponíveis em `acacia::ml`

- Modelos de regressão: `LinearRegression`, `PolynomialRegression`, `RidgeRegression`
- Funções de perda: `MSE`, `MAE`, `BinaryCrossEntropy`, `CategoricalCrossEntropy`
- Métricas: `MSE`, `RMSE`, `MAE`, `R2`, `Accuracy`, `Precision`, `Recall`, `F1Score`
- Otimizadores: `SGD`, `SGDWithMomentum`, `Adam`, `RMSProp`

---

## 2) Computer Vision (cv)

### Exemplo: Filtro Gaussiano e detecção de bordas

```cpp
#include "acacia/cv/filters.hpp"
#include "acacia/cv/kernel.hpp"

int main() {
    std::vector<std::vector<double>> image = {
        {0, 0, 0, 0, 0},
        {0, 0, 1, 0, 0},
        {0, 1, 1, 1, 0},
        {0, 0, 1, 0, 0},
        {0, 0, 0, 0, 0}
    };

    auto blurred = acacia::cv::gaussian_blur(image, 3, 1.0);
    auto edges = acacia::cv::sobel_edge_detection(image);

    (void)blurred;
    (void)edges;
    return 0;
}
```

### Recursos disponíveis em `acacia::cv`

- Kernels: `gaussian`, `sobel`, `laplacian`, `sharpen`, `box blur`
- Filtros: convolução, gaussiana, detecção de bordas (Sobel, Laplacian), filtro de média, filtro bilateral

---

## 3) Estatística (stats)

### Exemplo: Distribuição Normal

```cpp
#include "acacia/stats/distribution.hpp"

int main() {
    acacia::stats::Normal dist(0.0, 1.0);

    double sample = dist.sample();
    auto samples = dist.sample(100);

    double pdf_val = dist.pdf(0.0);
    double cdf_val = dist.cdf(0.0);

    (void)sample;
    (void)samples;
    (void)pdf_val;
    (void)cdf_val;
    return 0;
}
```

### Recursos disponíveis em `acacia::stats`

- Distribuições: `Normal`, `Uniform`, `Bernoulli`, `Categorical`, `Exponential`
- Funções: `sample()`, `pdf()`, `cdf()`

---

## 4) Dataset (dataset)

### Leitura e escrita de CSV

```cpp
#include "acacia/dataset/reader.hpp"
#include "acacia/dataset/writer.hpp"

int main() {
    auto df = acacia::dataset::read_csv("data.csv");
    acacia::dataset::write_csv("out.csv", df);
    return 0;
}
```

> ⚠️ Observação: os leitores/escritores de Excel, Parquet e banco de dados são marcados como *placeholder* e podem não estar totalmente implementados.

---

## 5) Utilitários (utils)

### Exemplo: Álgebra Linear e Cálculo

```cpp
#include "acacia/utils/linalg.hpp"
#include "acacia/utils/calculus.hpp"

int main() {
    acacia::utils::Vector<double> v{1.0, 2.0, 3.0};
    auto norm = acacia::utils::norm(v);

    auto derivative = acacia::utils::numerical_derivative(
        [](double x) { return x*x; }, 2.0);

    (void)norm;
    (void)derivative;
    return 0;
}
```

### Recursos disponíveis em `acacia::utils`

- Vetores e matrizes com operações básicas
- Diferenciação numérica, integração e métodos de busca de raízes

---

## 6) Compilando seu aplicativo usando Acacia

### Exemplo simples com `g++`

```bash
g++ -std=c++23 -I/path/para/acacia/include -L/path/para/acacia/lib -lacacia my_app.cpp -o my_app
```

### Exemplo usando `xmake` (sugerido)

Crie um `xmake.lua` simples:

```lua
add_rules("mode.debug", "mode.release")

target("my_app")
    set_kind("binary")
    set_languages("c++23")
    add_includedirs("/path/para/acacia/include")
    add_links("acacia")
    add_linkdirs("/path/para/acacia/lib")
    add_files("main.cpp")
```

Em seguida:

```bash
xmake
xmake run my_app
```

---

## 7) Referência rápida de includes

- `#include "acacia/ml/regression.hpp"`
- `#include "acacia/ml/loss.hpp"`
- `#include "acacia/ml/metrics.hpp"`
- `#include "acacia/cv/filters.hpp"`
- `#include "acacia/cv/kernel.hpp"`
- `#include "acacia/stats/distribution.hpp"`
- `#include "acacia/dataset/reader.hpp"`
- `#include "acacia/dataset/writer.hpp"`
- `#include "acacia/utils/linalg.hpp"`
- `#include "acacia/utils/calculus.hpp"`
