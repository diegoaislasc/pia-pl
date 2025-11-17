# Documentación Técnica - Solver Simplex de 2 Fases

## Arquitectura del Sistema

### Stack Tecnológico

- **Python 3.10+**: Lenguaje principal
- **NumPy 2.2.6**: Operaciones matriciales y álgebra lineal
- **Pandas 2.3.3**: Estructuras de datos tabulares
- **Streamlit 1.51.0**: Framework de UI interactivo

### Estructura de Archivos

```
pia-pl/
├── solver.py          # Clase TwoPhaseSimplexSolver
├── app.py             # Interfaz Streamlit
├── requirements.txt   # Dependencias
├── README.md          # Documentación de usuario
└── TechDocs.md        # Este archivo
```

## Clase TwoPhaseSimplexSolver

### Arquitectura OOP

La implementación utiliza Programación Orientada a Objetos con los siguientes principios:

- **Encapsulamiento**: Estado y comportamiento en una clase
- **Abstracción**: Métodos privados ocultan complejidad
- **Modularidad**: Separación clara de responsabilidades

### Atributos Principales

```python
# Entrada del problema
objective: np.ndarray      # Coeficientes función objetivo
A: np.ndarray             # Matriz de restricciones
b: np.ndarray             # Vector RHS
constraint_types: List    # ['<=', '>=', '=']
problem_type: str         # 'max' o 'min'

# Estado del solver
n_vars: int               # Número de variables originales
n_constraints: int        # Número de restricciones
n_slack: int              # Contador de variables de holgura
n_surplus: int            # Contador de variables de excedente
n_artificial: int         # Contador de variables artificiales

# Tabla Simplex
tableau: np.ndarray       # Tabla Simplex completa
basic_vars: List[int]     # Índices de variables básicas
var_names: List[str]      # Nombres de variables

# Índices especiales
artificial_var_indices: List[int]  # Índices de artificiales
surplus_var_indices: List[int]     # Índices de excedentes

# Resultado
status: str               # 'optimal', 'infeasible', 'unbounded'
optimal_value: float      # Valor óptimo de Z
solution: Dict            # Solución final
```

## Método Simplex de 2 Fases

### Fase 1: Encontrar Solución Básica Factible

**Objetivo**: Minimizar r = Σ(variables artificiales)

**Proceso**:

1. **Estandarización**:
   - Restricción `<=`: Agregar variable de holgura `s`
   - Restricción `>=`: Agregar excedente `-e` y artificial `+a`
   - Restricción `=`: Agregar artificial `+a`

2. **Inicialización**:
   - Variables artificiales son básicas inicialmente
   - Ajustar fila Phase1: restar filas donde hay artificiales básicas

3. **Iteración**:
   ```python
   while not optimal:
       entering = _find_entering_variable_phase1()  # Costo reducido más negativo
       leaving = _find_leaving_variable(entering)    # Mínimo ratio
       _pivot(leaving, entering)                     # Operación de pivoteo
   ```

4. **Criterio de Optimalidad**:
   - Todos los costos reducidos ≥ 0 (minimización)

5. **Resultados**:
   - `r* = 0`: Solución básica factible encontrada → Fase 2
   - `r* > 0`: Problema infactible

### Fase 2: Optimizar Función Objetivo

**Objetivo**: Optimizar Z = c₁x₁ + c₂x₂ + ... + cₙxₙ

**Preparación**:
1. Eliminar columnas de variables artificiales
2. Mantener fila Z ajustada de Fase 1
3. Eliminar fila Phase1

**Iteración**:
```python
while not optimal:
    entering = _find_entering_variable_phase2()  # Para MAX: más negativo
    leaving = _find_leaving_variable(entering)    # Mínimo ratio
    _pivot(leaving, entering)                     # Pivoteo
```

**Criterio de Optimalidad**:
- MAX: Todos los costos reducidos ≥ 0
- MIN: Todos los costos reducidos ≥ 0

## Algoritmos Clave

### Selección de Variable Entrante

**Fase 1** (Minimización):
```python
def _find_entering_variable_phase1():
    reduced_costs = tableau[1, :-1]  # Fila Phase1
    min_cost = infinity
    entering_col = None
    
    for j in range(len(reduced_costs)):
        if j in surplus_var_indices:
            continue  # Excluir excedentes
        
        if reduced_costs[j] < min_cost and reduced_costs[j] < -1e-10:
            min_cost = reduced_costs[j]
            entering_col = j
    
    return entering_col
```

**Fase 2**:
- MAX: Buscar costo reducido más negativo
- MIN: Buscar costo reducido más negativo

### Selección de Variable Saliente

Regla de mínimo ratio:
```python
def _find_leaving_variable(entering_col, phase):
    min_ratio = infinity
    leaving_row = None
    
    for i in range(n_constraints):
        pivot_element = tableau[row, entering_col]
        rhs = tableau[row, -1]
        
        if pivot_element > 1e-10:
            ratio = rhs / pivot_element
            if ratio >= 0 and ratio < min_ratio:
                min_ratio = ratio
                leaving_row = i
    
    return leaving_row, (leaving_row is None)
```

### Operación de Pivoteo

```python
def _pivot(pivot_row, pivot_col):
    # 1. Normalizar fila pivote
    tableau[actual_row, :] /= pivot_element
    
    # 2. Hacer cero los demás elementos
    for i in range(tableau.shape[0]):
        if i != actual_row:
            multiplier = tableau[i, pivot_col]
            tableau[i, :] -= multiplier * tableau[actual_row, :]
    
    # 3. Actualizar variable básica
    basic_vars[pivot_row] = pivot_col
```

## Estructura de la Tabla Simplex

### Fase 1

```
        x₁   x₂  ...  xₙ   s₁  ...  e₁   a₁  ...  RHS
Z     | c₁   c₂  ... cₙ   0   ...  0    0   ...   0  |
Phase1| 0    0   ...  0   0   ...  1    1   ...   M  |
R₁    | a₁₁  a₁₂ ... a₁ₙ  1   ...  0    0   ...   b₁ |
R₂    | a₂₁  a₂₂ ... a₂ₙ  0   ... -1    1   ...   b₂ |
...   | ...  ... ... ...  ... ...  ...  ... ...  ...  |
```

### Fase 2

```
        x₁   x₂  ...  xₙ   s₁  ...  e₁  ...  RHS
Z     | r₁   r₂  ... rₙ   r₁  ...  r₁  ...   Z*  |
R₁    | a₁₁  a₁₂ ... a₁ₙ  a₁  ...  a₁  ...   b₁  |
R₂    | a₂₁  a₂₂ ... a₂ₙ  a₂  ...  a₂  ...   b₂  |
...   | ...  ... ... ...  ... ...  ...  ...  ...  |
```

## Complejidad Computacional

- **Peor caso**: O(2^n) iteraciones
- **Caso promedio**: O(m×n) por iteración
- **Espacio**: O(m×n) para la tabla

Donde:
- n = número de variables
- m = número de restricciones

## Manejo de Casos Especiales

### RHS Negativos

Si `b[i] < 0`:
1. Multiplicar restricción por -1
2. Invertir signo de desigualdad
3. Mantener igualdades

### Degeneración

Variables artificiales en la base con valor 0:
- Detectada al final de Fase 1
- Status: 'degenerate'
- Solución puede no ser única

### Problema Infactible

Fase 1 termina con `r* > 0`:
- No existe solución que satisfaga todas las restricciones
- Status: 'infeasible'

### Problema No Acotado

Columna entrante no tiene ratios positivos:
- Función objetivo puede crecer indefinidamente
- Status: 'unbounded'

## Uso de Generadores

La función `solve()` es un generador que yield información en cada iteración:

```python
def solve() -> Generator[Dict, None, Dict]:
    for iteration_info in phase1:
        yield iteration_info
    
    prepare_phase2()
    
    for iteration_info in phase2:
        yield iteration_info
    
    return final_result
```

**Ventajas**:
- Visualización paso a paso en la UI
- Bajo consumo de memoria
- Evaluación lazy

## Validaciones

### Entrada

1. Dimensiones consistentes de A, b, constraint_types
2. Al menos una variable en la función objetivo
3. Problem_type válido ('max' o 'min')
4. RHS no negativos (ajuste automático)

### Durante Ejecución

1. Variables básicas válidas en cada iteración
2. Elementos pivote > 0
3. Límite de iteraciones (1000)

## Formato de Salida

```python
{
    'status': 'optimal' | 'infeasible' | 'unbounded' | 'error',
    'solution': {
        'x1': float,
        'x2': float,
        ...
    },
    'optimal_value': float,
    'message': str
}
```

## Ejemplo de Uso Programático

```python
from solver import TwoPhaseSimplexSolver

# Definir problema
solver = TwoPhaseSimplexSolver(
    objective=[40, 30, 20],
    A=[[1, 1, 1], [2, 1, -1]],
    b=[10, 4],
    constraint_types=['=', '>='],
    problem_type='max'
)

# Resolver
for iteration_info in solver.solve():
    phase = iteration_info['phase']
    iteration = iteration_info['iteration']
    print(f"Fase {phase}, Iteración {iteration}")
    print(iteration_info['tableau'])

# Obtener resultado
if solver.status == 'optimal':
    solution = solver._extract_solution()
    print(f"Solución: {solution}")
    print(f"Valor óptimo: {solver.optimal_value}")
```

## Testing

El solver ha sido validado con:

1. **Problema de prueba**:
   - MAX Z = 40x₁ + 30x₂ + 20x₃
   - x₁ + x₂ + x₃ = 10
   - 2x₁ + x₂ - x₃ ≥ 4
   - **Resultado**: x₁=10, x₂=0, x₃=0, Z*=400 ✓

2. **Casos especiales**:
   - Problemas infactibles
   - Problemas no acotados
   - Degeneración

## Extensiones Futuras

- Método Simplex Revisado
- Simplex Dual
- Branch & Bound para PL entera
- Análisis de sensibilidad
- Presolvers
- Paralelización

## Referencias

- Hillier & Lieberman: "Introducción a la Investigación de Operaciones"
- Bazaraa, Jarvis & Sherali: "Linear Programming and Network Flows"
- NumPy Documentation: https://numpy.org/doc/
- Streamlit Documentation: https://docs.streamlit.io/

