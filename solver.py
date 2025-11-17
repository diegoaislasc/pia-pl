"""
Solver Simplex de 2 Fases para Programación Lineal
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Generator


class TwoPhaseSimplexSolver:
    """
    Solver para problemas de Programación Lineal usando el Método Simplex de 2 Fases.
    """
    
    def __init__(self, objective: List[float], A: List[List[float]], 
                 b: List[float], constraint_types: List[str], 
                 problem_type: str = 'max'):
        """
        Inicializa el solver con el problema de PL.
        
        Args:
            objective: Coeficientes de la función objetivo
            A: Matriz de coeficientes de las restricciones
            b: Vector de valores del lado derecho
            constraint_types: Tipos de restricción ['<=', '>=', '=']
            problem_type: 'max' o 'min'
        """
        self.objective = np.array(objective, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.constraint_types = constraint_types
        self.problem_type = problem_type.lower()
        
        self._validate_inputs()
        
        self.n_vars = len(objective)
        self.n_constraints = len(b)
        self.n_slack = 0
        self.n_surplus = 0
        self.n_artificial = 0
        
        self.var_names = []
        self.basic_vars = []
        self.tableau = None
        
        self.phase1_history = []
        self.phase2_history = []
        
        self.status = 'pending'
        self.solution = {}
        self.optimal_value = None
        
        self._standardize_problem()
        
    def _validate_inputs(self):
        """Valida dimensiones y ajusta RHS negativos."""
        if len(self.objective) == 0:
            raise ValueError("La función objetivo no puede estar vacía")
        
        if len(self.A) != len(self.b):
            raise ValueError(f"Dimensiones inconsistentes: A tiene {len(self.A)} filas pero b tiene {len(self.b)} elementos")
        
        if len(self.A) != len(self.constraint_types):
            raise ValueError(f"Debe haber un tipo de restricción para cada fila de A")
        
        for i, row in enumerate(self.A):
            if len(row) != len(self.objective):
                raise ValueError(f"Fila {i+1} de A tiene {len(row)} columnas, pero se esperan {len(self.objective)}")
        
        if self.problem_type not in ['max', 'min']:
            raise ValueError(f"Tipo de problema debe ser 'max' o 'min', se recibió '{self.problem_type}'")
        
        for i in range(len(self.b)):
            if self.b[i] < 0:
                self.b[i] = -self.b[i]
                self.A[i] = [-coeff for coeff in self.A[i]]
                if self.constraint_types[i] == '<=':
                    self.constraint_types[i] = '>='
                elif self.constraint_types[i] == '>=':
                    self.constraint_types[i] = '<='
    
    def _standardize_problem(self):
        """Convierte el problema a forma estándar agregando variables auxiliares."""
        self.var_names = [f'x{i+1}' for i in range(self.n_vars)]
        
        self.artificial_var_indices = []
        self.surplus_var_indices = []
        
        slack_count = 0
        surplus_count = 0
        artificial_count = 0
        
        for i, constraint_type in enumerate(self.constraint_types):
            if constraint_type == '<=':
                self.n_slack += 1
                slack_count += 1
                self.var_names.append(f's{slack_count}')
            elif constraint_type == '>=':
                self.n_surplus += 1
                self.n_artificial += 1
                surplus_count += 1
                artificial_count += 1
                self.var_names.append(f'e{surplus_count}')
                self.var_names.append(f'a{artificial_count}')
            elif constraint_type == '=':
                self.n_artificial += 1
                artificial_count += 1
                self.var_names.append(f'a{artificial_count}')
        
        total_vars = self.n_vars + self.n_slack + self.n_surplus + self.n_artificial
        A_std = np.zeros((self.n_constraints, total_vars))
        A_std[:, :self.n_vars] = self.A
        
        col_idx = self.n_vars
        for i, constraint_type in enumerate(self.constraint_types):
            if constraint_type == '<=':
                A_std[i, col_idx] = 1.0
                col_idx += 1
            elif constraint_type == '>=':
                A_std[i, col_idx] = -1.0
                self.surplus_var_indices.append(col_idx)
                col_idx += 1
                A_std[i, col_idx] = 1.0
                self.artificial_var_indices.append(col_idx)
                col_idx += 1
            elif constraint_type == '=':
                A_std[i, col_idx] = 1.0
                self.artificial_var_indices.append(col_idx)
                col_idx += 1
        
        tableau_rows = 2 + self.n_constraints
        tableau_cols = total_vars + 1
        self.tableau = np.zeros((tableau_rows, tableau_cols))
        
        if self.problem_type == 'max':
            self.tableau[0, :self.n_vars] = -self.objective
        else:
            self.tableau[0, :self.n_vars] = self.objective
        
        for artificial_idx in self.artificial_var_indices:
            self.tableau[1, artificial_idx] = 1.0
        
        self.tableau[2:, :total_vars] = A_std
        self.tableau[2:, total_vars] = self.b
        
        self.basic_vars = []
        constraint_to_basic = {}
        
        for i in range(self.n_constraints):
            constraint_row = i + 2
            found_basic = False
            
            for j in range(total_vars):
                if abs(self.tableau[constraint_row, j] - 1.0) < 1e-10:
                    is_basic = True
                    for k in range(self.n_constraints):
                        if k != i and abs(self.tableau[k+2, j]) > 1e-10:
                            is_basic = False
                            break
                    
                    if is_basic:
                        constraint_to_basic[i] = j
                        found_basic = True
                        break
            
            if not found_basic:
                raise ValueError(f"No se pudo encontrar variable básica para restricción {i+1}")
        
        for i in range(self.n_constraints):
            self.basic_vars.append(constraint_to_basic[i])
        
        self._create_phase1_objective()
    
    def _create_phase1_objective(self):
        """Crea la función objetivo auxiliar para Fase 1."""
        for i, basic_var_idx in enumerate(self.basic_vars):
            if basic_var_idx in self.artificial_var_indices:
                constraint_row = i + 2
                self.tableau[1, :] -= self.tableau[constraint_row, :]
    
    def _get_variable_names(self) -> List[str]:
        """Retorna la lista de nombres de variables."""
        return self.var_names.copy()
    
    def _format_table(self, tableau: np.ndarray = None) -> np.ndarray:
        """Formatea la tabla para visualización."""
        if tableau is None:
            tableau = self.tableau.copy()
        return np.round(tableau, decimals=4)
    
    def _find_entering_variable_phase1(self) -> Optional[int]:
        """Encuentra la variable entrante para Fase 1."""
        reduced_costs = self.tableau[1, :-1]
        
        min_cost = np.inf
        entering_col = None
        
        for j in range(len(reduced_costs)):
            if j in self.surplus_var_indices:
                continue
            
            if reduced_costs[j] < min_cost and reduced_costs[j] < -1e-10:
                min_cost = reduced_costs[j]
                entering_col = j
        
        return entering_col
    
    def _find_entering_variable_phase2(self) -> Optional[int]:
        """Encuentra la variable entrante para Fase 2."""
        reduced_costs = self.tableau[0, :-1]
        artificial_start = self.n_vars + self.n_slack + self.n_surplus
        total_vars = len(self.var_names)
        
        entering_col = None
        
        if self.problem_type == 'max':
            min_cost = np.inf
            for j in range(min(len(reduced_costs), total_vars)):
                if j < artificial_start and reduced_costs[j] < min_cost and reduced_costs[j] < -1e-10:
                    min_cost = reduced_costs[j]
                    entering_col = j
        else:
            min_cost = np.inf
            for j in range(min(len(reduced_costs), total_vars)):
                if j < artificial_start and reduced_costs[j] < min_cost and reduced_costs[j] < -1e-10:
                    min_cost = reduced_costs[j]
                    entering_col = j
        
        return entering_col
    
    def _find_leaving_variable(self, entering_col: int, phase: int = 1) -> Tuple[Optional[int], bool]:
        """Encuentra la variable saliente usando la regla de mínimo ratio."""
        min_ratio = np.inf
        leaving_row = None
        is_unbounded = True
        
        start_idx = 2 if phase == 1 else 1
        
        for i in range(self.n_constraints):
            constraint_row = i + start_idx
            pivot_element = self.tableau[constraint_row, entering_col]
            rhs = self.tableau[constraint_row, -1]
            
            if pivot_element > 1e-10:
                is_unbounded = False
                ratio = rhs / pivot_element
                if ratio >= 0 and ratio < min_ratio:
                    min_ratio = ratio
                    leaving_row = i
        
        return leaving_row, is_unbounded
    
    def _pivot(self, pivot_row: int, pivot_col: int):
        """Realiza la operación de pivoteo."""
        actual_row = pivot_row + 2
        pivot_element = self.tableau[actual_row, pivot_col]
        
        self.tableau[actual_row, :] /= pivot_element
        
        for i in range(self.tableau.shape[0]):
            if i != actual_row:
                multiplier = self.tableau[i, pivot_col]
                self.tableau[i, :] -= multiplier * self.tableau[actual_row, :]
        
        self.basic_vars[pivot_row] = pivot_col
    
    def _check_phase1_optimality(self) -> Tuple[bool, str]:
        """Verifica optimalidad en Fase 1."""
        reduced_costs = self.tableau[1, :-1]
        
        if np.all(reduced_costs >= -1e-10):
            phase1_value = -self.tableau[1, -1]
            
            if phase1_value > 1e-10:
                return True, 'infeasible'
            else:
                return True, 'optimal'
        
        return False, 'continue'
    
    def _check_phase2_optimality(self) -> bool:
        """Verifica optimalidad en Fase 2."""
        reduced_costs = self.tableau[0, :-1]
        artificial_start = self.n_vars + self.n_slack + self.n_surplus
        relevant_costs = reduced_costs[:artificial_start]
        
        if self.problem_type == 'max':
            return np.all(relevant_costs >= -1e-10)
        else:
            return np.all(relevant_costs >= -1e-10)
    
    def _solve_phase_1(self) -> Generator[Dict, None, str]:
        """Resuelve la Fase 1 del método Simplex."""
        iteration = 0
        max_iterations = 1000
        
        yield {
            'phase': 1,
            'iteration': iteration,
            'tableau': self.tableau.copy(),
            'basic_vars': self.basic_vars.copy(),
            'entering': None,
            'leaving': None,
            'pivot': None
        }
        
        while iteration < max_iterations:
            iteration += 1
            
            is_optimal, status = self._check_phase1_optimality()
            if is_optimal:
                if status == 'infeasible':
                    self.status = 'infeasible'
                    return 'infeasible'
                else:
                    break
            
            entering_col = self._find_entering_variable_phase1()
            if entering_col is None:
                break
            
            leaving_row, is_unbounded = self._find_leaving_variable(entering_col)
            
            if is_unbounded or leaving_row is None:
                self.status = 'unbounded'
                return 'unbounded'
            
            pivot_value = self.tableau[leaving_row + 2, entering_col]
            self._pivot(leaving_row, entering_col)
            
            yield {
                'phase': 1,
                'iteration': iteration,
                'tableau': self.tableau.copy(),
                'basic_vars': self.basic_vars.copy(),
                'entering': entering_col,
                'leaving': leaving_row,
                'pivot': pivot_value
            }
        
        if iteration >= max_iterations:
            self.status = 'error'
            return 'error'
        
        has_degenerate_artificial = False
        for i, basic_var_idx in enumerate(self.basic_vars):
            if basic_var_idx in self.artificial_var_indices:
                rhs_value = self.tableau[i + 2, -1]
                if abs(rhs_value) < 1e-10:
                    has_degenerate_artificial = True
                    break
        
        if has_degenerate_artificial:
            self.status = 'degenerate'
        
        return 'optimal'
    
    def _prepare_phase2(self):
        """Prepara la tabla para Fase 2."""
        total_vars_with_artificials = self.n_vars + self.n_slack + self.n_surplus + self.n_artificial
        total_vars_original = self.n_vars + self.n_slack + self.n_surplus
        
        non_artificial_cols = [j for j in range(total_vars_with_artificials) 
                               if j not in self.artificial_var_indices]
        
        new_tableau = np.zeros((1 + self.n_constraints, total_vars_original + 1))
        
        for new_col, old_col in enumerate(non_artificial_cols):
            new_tableau[0, new_col] = self.tableau[0, old_col]
        new_tableau[0, -1] = self.tableau[0, -1]
        
        for new_col, old_col in enumerate(non_artificial_cols):
            new_tableau[1:, new_col] = self.tableau[2:, old_col]
        new_tableau[1:, -1] = self.tableau[2:, -1]
        
        self.tableau = new_tableau
        
        old_to_new_idx = {}
        new_idx = 0
        for old_idx in range(total_vars_with_artificials):
            if old_idx not in self.artificial_var_indices:
                old_to_new_idx[old_idx] = new_idx
                new_idx += 1
        
        new_basic_vars = []
        for basic_var_idx in self.basic_vars:
            if basic_var_idx not in self.artificial_var_indices:
                mapped_idx = old_to_new_idx[basic_var_idx]
                new_basic_vars.append(mapped_idx)
            else:
                new_basic_vars.append(-1)
        
        self.basic_vars = new_basic_vars
        
        new_var_names = []
        for i, name in enumerate(self.var_names):
            if i not in self.artificial_var_indices:
                new_var_names.append(name)
        self.var_names = new_var_names
    
    def _solve_phase_2(self) -> Generator[Dict, None, str]:
        """Resuelve la Fase 2 del método Simplex."""
        iteration = 0
        max_iterations = 1000
        
        yield {
            'phase': 2,
            'iteration': iteration,
            'tableau': self.tableau.copy(),
            'basic_vars': self.basic_vars.copy(),
            'entering': None,
            'leaving': None,
            'pivot': None
        }
        
        while iteration < max_iterations:
            iteration += 1
            
            if self._check_phase2_optimality():
                self.status = 'optimal'
                break
            
            entering_col = self._find_entering_variable_phase2()
            if entering_col is None:
                self.status = 'optimal'
                break
            
            leaving_row, is_unbounded = self._find_leaving_variable(entering_col, phase=2)
            
            if is_unbounded or leaving_row is None:
                self.status = 'unbounded'
                return 'unbounded'
            
            pivot_value = self.tableau[leaving_row + 1, entering_col]
            
            actual_row = leaving_row + 1
            pivot_element = self.tableau[actual_row, entering_col]
            
            self.tableau[actual_row, :] /= pivot_element
            
            for i in range(self.tableau.shape[0]):
                if i != actual_row:
                    multiplier = self.tableau[i, entering_col]
                    self.tableau[i, :] -= multiplier * self.tableau[actual_row, :]
            
            self.basic_vars[leaving_row] = entering_col
            
            yield {
                'phase': 2,
                'iteration': iteration,
                'tableau': self.tableau.copy(),
                'basic_vars': self.basic_vars.copy(),
                'entering': entering_col,
                'leaving': leaving_row,
                'pivot': pivot_value
            }
        
        if iteration >= max_iterations:
            self.status = 'error'
            return 'error'
        
        return 'optimal'
    
    def _extract_solution(self) -> Dict:
        """Extrae la solución del problema desde la tabla final."""
        solution = {}
        
        for var_name in self.var_names:
            solution[var_name] = 0.0
        
        for i, basic_var_idx in enumerate(self.basic_vars):
            if basic_var_idx < len(self.var_names):
                constraint_row = i + 1
                rhs_value = self.tableau[constraint_row, -1]
                solution[self.var_names[basic_var_idx]] = rhs_value
        
        z_value = self.tableau[0, -1]
        self.optimal_value = z_value
        
        return solution
    
    def solve(self) -> Generator[Dict, None, Dict]:
        """
        Método principal que orquesta la solución del problema.
        Ejecuta Fase 1 y luego Fase 2 si es factible.
        """
        phase1_gen = self._solve_phase_1()
        for iteration_info in phase1_gen:
            self.phase1_history.append(iteration_info)
            yield iteration_info
        
        if self.status == 'infeasible':
            return {
                'status': 'infeasible',
                'solution': {},
                'optimal_value': None,
                'message': 'El problema es infactible'
            }
        
        if self.status == 'unbounded':
            return {
                'status': 'unbounded',
                'solution': {},
                'optimal_value': None,
                'message': 'El problema es no acotado'
            }
        
        self._prepare_phase2()
        
        phase2_gen = self._solve_phase_2()
        for iteration_info in phase2_gen:
            self.phase2_history.append(iteration_info)
            yield iteration_info
        
        if self.status == 'optimal':
            solution = self._extract_solution()
            return {
                'status': 'optimal',
                'solution': solution,
                'optimal_value': self.optimal_value,
                'message': 'Solución óptima encontrada'
            }
        elif self.status == 'unbounded':
            return {
                'status': 'unbounded',
                'solution': {},
                'optimal_value': None,
                'message': 'El problema es no acotado'
            }
        else:
            return {
                'status': 'error',
                'solution': {},
                'optimal_value': None,
                'message': 'Error en el proceso de solución'
            }
