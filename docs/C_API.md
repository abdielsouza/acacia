# C API (Bindings)

A Acacia também expõe uma **C API mínima** que facilita a criação de bindings para outras linguagens (Python, Rust, Go, etc.) e o uso em projetos C.

## 1) Header principal

A API C está disponível em:

- `include/acacia/c_api.h`

Ela é exportada pelo mesmo binário/ biblioteca compartilhada `libacacia.so` (ou `.dll` no Windows).

## 2) Exemplo de uso em C

```c
#include <stdio.h>
#include "acacia/c_api.h"

int main(void) {
    // Exemplo básico de regressão linear em C
    double X[5][1] = {{1}, {2}, {3}, {4}, {5}};
    double y[5] = {2, 4, 6, 8, 10};

    acacia_lr_t* model = acacia_lr_create();
    if (!model) {
        fprintf(stderr, "Erro: %s\n", acacia_get_last_error());
        return 1;
    }

    if (acacia_lr_fit(model, &X[0][0], 5, 1, y) != ACACIA_STATUS_OK) {
        fprintf(stderr, "Erro: %s\n", acacia_get_last_error());
        acacia_lr_destroy(model);
        return 1;
    }

    double predictions[5];
    if (acacia_lr_predict(model, &X[0][0], 5, 1, predictions) != ACACIA_STATUS_OK) {
        fprintf(stderr, "Erro: %s\n", acacia_get_last_error());
        acacia_lr_destroy(model);
        return 1;
    }

    for (int i = 0; i < 5; ++i) {
        printf("pred[%d] = %f\n", i, predictions[i]);
    }

    acacia_lr_destroy(model);
    return 0;
}
```

## 3) Linkando a biblioteca

Exemplo de compilação e linkagem com `gcc`:

```bash
gcc -std=c11 -I/path/para/acacia/include -L/path/para/acacia/lib -lacacia -o my_app my_app.c
```

> Substitua `-L/path/para/acacia/lib` pelo caminho onde a biblioteca `libacacia.*` foi instalada.
