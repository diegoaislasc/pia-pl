# Solver Simplex de 2 Fases

Aplicación interactiva para resolver problemas de Programación Lineal usando el Método Simplex de las 2 Fases.

## Características

- Solución mediante el Método de las Dos Fases
- Visualización paso a paso de ambas fases
- Interfaz interactiva con Streamlit
- Soporte para maximización y minimización
- Manejo de restricciones `<=`, `>=` y `=`
- Detección de casos especiales: infactible, no acotado

## Instalación

```bash
pip3 install --user numpy pandas streamlit
```

## Ejecución

```bash
streamlit run app.py
```

O con ruta completa:
```bash
/Users/diegoandre/Library/Python/3.10/bin/streamlit run app.py
```

## Uso

### Configuración (Sidebar)
1. Selecciona tipo de problema (Maximizar/Minimizar)
2. Ingresa número de variables
3. Ingresa número de restricciones

### Entrada de Datos (Pantalla Principal)
1. Ingresa coeficientes de la función objetivo
2. Para cada restricción:
   - Coeficientes de las variables
   - Tipo de desigualdad (`<=`, `>=`, `=`)
   - Valor del lado derecho (RHS)
3. Haz clic en "Resolver"

## Ejemplo

**Problema:**
```
MAX Z = 40x₁ + 30x₂ + 20x₃
s.a.
  x₁ + x₂ + x₃ = 10
  2x₁ + x₂ - x₃ >= 4
  x₁, x₂, x₃ >= 0
```

**Solución:**
```
x₁ = 10
x₂ = 0
x₃ = 0
Z* = 400
```

## Método de las 2 Fases

### Fase 1
- **Objetivo**: Encontrar solución básica factible
- **Método**: Minimizar suma de variables artificiales
- **Resultado**: Si mínimo = 0 → factible, si > 0 → infactible

### Fase 2
- **Objetivo**: Optimizar función objetivo original
- **Método**: Simplex estándar desde solución de Fase 1
- **Resultado**: Solución óptima, no acotada, o degenerada

## Estructura del Proyecto

```
pia-pl/
├── solver.py          # Clase TwoPhaseSimplexSolver
├── app.py             # Interfaz Streamlit
├── requirements.txt   # Dependencias
├── README.md          # Este archivo
└── TechDocs.md        # Documentación técnica
```

## Documentación

- **README.md**: Guía de usuario (este archivo)
- **TechDocs.md**: Documentación técnica detallada

## Tecnologías

- Python 3.10+
- NumPy 2.2.6: Operaciones matriciales
- Pandas 2.3.3: Estructuras de datos
- Streamlit 1.51.0: Framework UI

## Estado

Completamente funcional y probado con problemas de:
- Maximización y minimización
- Restricciones mixtas
- Casos especiales

## Licencia

Proyecto académico - Programación Lineal
