# Instalação

Este documento descreve como instalar e preparar o ambiente para compilar e usar a biblioteca **Acacia**.

---

## Requisitos

- **Sistema operacional:** Linux (ou qualquer sistema suportado pelo xmake).
- **Compilador C++** compatível com **C++23**:
  - GCC 11+ (ex: `g++ --version`)
  - Clang 14+
  - MSVC 2022+ (para Windows)
- **[xmake](https://xmake.io)** (gerenciador de build usado pelo projeto)

> ⚡ Dica: você pode instalar o xmake via `brew install xmake`, `pip install xmake`, ou seguindo as instruções oficiais em https://xmake.io.

---

## 1) Clonar o repositório

```bash
git clone https://github.com/<seu-usuario>/acacia.git
cd acacia
```

> Substitua `https://github.com/<seu-usuario>/acacia.git` pela URL real do repositório, se necessário.

---

## 2) Compilar a biblioteca

O projeto já inclui um `xmake.lua` configurado para gerar a biblioteca compartilhada `acacia` e o executável de testes `acacia-tests`.

### 2.1 Compilação padrão (modo Release)

```bash
xmake
```

### 2.2 Compilação em modo Debug

```bash
xmake -m debug
```

### 2.3 Configurações avançadas

```bash
# Alterar o diretório de saída
xmake f -o build/out
xmake

# Especificar plataforma/arquitetura
xmake f -p linux -a x86_64 -m release
xmake
```

---

## 3) Executar os testes

```bash
xmake run acacia-tests
```

---

## 4) Instalar a biblioteca no sistema

Por padrão, o `xmake install` copia a biblioteca e headers para os diretórios de instalação do sistema.

```bash
xmake install
```

Se quiser instalar em um diretório específico (por exemplo, `~/local`):

```bash
xmake install -o ~/local
```

---

## 5) Usar a biblioteca em outro projeto

Após compilar/instalar, basta incluir os headers em seus arquivos fonte e linkar contra a biblioteca `acacia`.

### Incluindo cabeçalhos (exemplo)

```cpp
#include "acacia/ml/regression.hpp"
#include "acacia/cv/filters.hpp"
```

### Linkagem (exemplo com g++)

```bash
g++ -std=c++23 -I/path/para/acacia/include -L/path/para/acacia/lib -lacacia myapp.cpp -o myapp
```

> Se você instalou a biblioteca usando `xmake install`, substitua `/path/para/acacia` por `/usr/local` ou pelo diretório de instalação escolhido.
