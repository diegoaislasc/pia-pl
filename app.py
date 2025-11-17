"""
Aplicación Streamlit para resolver problemas de Programación Lineal
usando el Método Simplex de las 2 Fases.
"""

import streamlit as st
import numpy as np
import pandas as pd
from solver import TwoPhaseSimplexSolver


def format_problem_latex(objective: list, A: list, b: list, 
                         constraint_types: list, problem_type: str) -> str:
    """Formatea el problema de PL en formato LaTeX."""
    obj_str = "\\text{" + problem_type.capitalize() + "} \\quad Z = "
    obj_terms = []
    for i, coeff in enumerate(objective):
        if abs(coeff) > 1e-10:
            if i == 0:
                obj_terms.append(f"{coeff:.2f}x_{{{i+1}}}")
            else:
                sign = "+" if coeff >= 0 else ""
                obj_terms.append(f"{sign}{coeff:.2f}x_{{{i+1}}}")
    obj_str += " + ".join(obj_terms).replace("+ -", "- ")
    
    constraints_str = "\\\\\n\\text{Sujeto a:} \\\\\n"
    constraint_lines = []
    for i, (row, rhs, ctype) in enumerate(zip(A, b, constraint_types)):
        terms = []
        for j, coeff in enumerate(row):
            if abs(coeff) > 1e-10:
                if j == 0:
                    terms.append(f"{coeff:.2f}x_{{{j+1}}}")
                else:
                    sign = "+" if coeff >= 0 else ""
                    terms.append(f"{sign}{coeff:.2f}x_{{{j+1}}}")
        constraint_line = " + ".join(terms).replace("+ -", "- ") + f" {ctype} {rhs:.2f}"
        constraint_lines.append(constraint_line)
    
    constraints_str += " \\\\\n".join(constraint_lines)
    
    var_count = len(objective)
    non_neg_str = "\\\\\nx_1, x_2"
    if var_count > 2:
        non_neg_str += ", \\ldots"
    non_neg_str += f", x_{{{var_count}}} \\geq 0"
    
    return f"\\begin{{aligned}}\n{obj_str}{constraints_str}{non_neg_str}\n\\end{{aligned}}"


def create_tableau_dataframe(tableau: np.ndarray, var_names: list, 
                             phase: int, basic_vars: list = None) -> pd.DataFrame:
    """Crea un DataFrame de pandas para visualizar la tabla Simplex."""
    n_vars = len(var_names)
    n_rows = tableau.shape[0]
    
    if phase == 1:
        row_names = ['Z', 'Phase 1']
        row_names.extend([f'R{i+1}' for i in range(n_rows - 2)])
    else:
        row_names = ['Z']
        row_names.extend([f'R{i+1}' for i in range(n_rows - 1)])
    
    columns = var_names + ['RHS']
    df = pd.DataFrame(tableau, index=row_names, columns=columns)
    
    if basic_vars is not None:
        basic_col = []
        for i, row_name in enumerate(row_names):
            if i == 0:
                basic_col.append('')
            elif phase == 1 and i == 1:
                basic_col.append('')
            else:
                constraint_idx = i - 1 if phase == 2 else i - 2
                if constraint_idx < len(basic_vars):
                    var_idx = basic_vars[constraint_idx]
                    if var_idx < len(var_names):
                        basic_col.append(var_names[var_idx])
                    else:
                        basic_col.append('')
                else:
                    basic_col.append('')
        
        df.insert(0, 'VB', basic_col)
    
    return df


def main():
    """Función principal de la aplicación Streamlit."""
    
    st.set_page_config(
        page_title="Solver Simplex 2 Fases",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("Solver Simplex de 2 Fases")
    st.markdown("""
    Resuelve problemas de Programación Lineal usando el **Método Simplex de las 2 Fases**.
    """)
    
    with st.sidebar:
        st.header("Configuración")
        
        problem_type = st.selectbox(
            "Tipo de problema",
            ['Maximizar', 'Minimizar'],
            index=0
        )
        problem_type_short = 'max' if problem_type == 'Maximizar' else 'min'
        
        n_vars = st.number_input(
            "Número de variables",
            min_value=1,
            max_value=20,
            value=2,
            step=1
        )
        
        n_constraints = st.number_input(
            "Número de restricciones",
            min_value=1,
            max_value=20,
            value=3,
            step=1
        )
        
        st.markdown("---")
        st.markdown("### Instrucciones")
        st.markdown("""
        1. Configura el problema
        2. Ingresa los coeficientes
        3. Selecciona los signos
        4. Haz clic en Resolver
        """)
    
    st.header("Entrada de Datos")
    
    st.subheader("Función Objetivo")
    st.markdown(f"Coeficientes para las {n_vars} variables:")
    
    obj_cols = st.columns(n_vars)
    objective_coeffs = []
    for i in range(n_vars):
        with obj_cols[i]:
            coeff = st.number_input(
                f"x{i+1}",
                value=0,
                step=1,
                key=f"obj_{i}",
                format="%d"
            )
            objective_coeffs.append(coeff)
    
    st.subheader("Restricciones")
    st.markdown(f"Coeficientes y signos para las {n_constraints} restricciones:")
    
    constraints = []
    constraint_types = []
    
    for i in range(n_constraints):
        with st.container():
            constraint_cols = st.columns(n_vars + 3)
            
            with constraint_cols[0]:
                st.markdown(f"**R{i+1}**")
                st.markdown("<br>", unsafe_allow_html=True)
            
            constraint_coeffs = []
            for j in range(n_vars):
                with constraint_cols[j + 1]:
                    coeff = st.number_input(
                        f"x{j+1}",
                        value=0,
                        step=1,
                        key=f"constraint_{i}_var_{j}",
                        format="%d",
                        label_visibility="visible"
                    )
                    constraint_coeffs.append(coeff)
            
            with constraint_cols[n_vars + 1]:
                st.markdown("<br>", unsafe_allow_html=True)
                constraint_type = st.selectbox(
                    "Signo",
                    ['<=', '>=', '='],
                    index=0,
                    key=f"sign_{i}",
                    label_visibility="visible"
                )
                constraint_types.append(constraint_type)
            
            with constraint_cols[n_vars + 2]:
                rhs_value = st.number_input(
                    "RHS",
                    value=0,
                    step=1,
                    key=f"rhs_{i}",
                    format="%d",
                    label_visibility="visible"
                )
            
            constraints.append({
                'coeffs': constraint_coeffs,
                'type': constraint_type,
                'rhs': rhs_value
            })
            
            if i < n_constraints - 1:
                st.markdown("---")
    
    solve_button = st.button("Resolver", type="primary", use_container_width=True)
    
    if solve_button:
        try:
            if all(c == 0 for c in objective_coeffs):
                st.error("La función objetivo debe tener al menos un coeficiente no cero")
                st.stop()
            
            objective = [float(c) for c in objective_coeffs]
            
            A = []
            b = []
            constraint_types_list = []
            
            for i, constraint in enumerate(constraints):
                coeffs = [float(c) for c in constraint['coeffs']]
                
                if all(c == 0 for c in coeffs):
                    st.warning(f"La restricción {i+1} tiene todos los coeficientes en cero. Se omitirá.")
                    continue
                
                A.append(coeffs)
                b.append(float(constraint['rhs']))
                constraint_types_list.append(constraint['type'])
            
            if len(A) == 0:
                st.error("Debe haber al menos una restricción válida")
                st.stop()
            
            if len(A[0]) != n_vars:
                st.error(f"Las restricciones deben tener {n_vars} coeficientes")
                st.stop()
            
            solver = TwoPhaseSimplexSolver(
                objective=objective,
                A=A,
                b=b,
                constraint_types=constraint_types_list,
                problem_type=problem_type_short
            )
            
            st.header("Problema Formulado")
            latex_problem = format_problem_latex(
                objective, A, b, constraint_types_list, problem_type_short
            )
            st.latex(latex_problem)
            
            st.header("Proceso de Solución")
            
            phase1_container = st.container()
            phase2_container = st.container()
            
            current_phase = None
            iteration_counters = {1: 0, 2: 0}
            
            result_gen = solver.solve()
            final_result = None
            
            try:
                for iteration_info in result_gen:
                    phase = iteration_info['phase']
                    iteration = iteration_info['iteration']
                    
                    if phase != current_phase:
                        current_phase = phase
                        if phase == 1:
                            phase1_container.header("Fase 1: Encontrar Solución Básica Factible")
                            phase1_container.markdown("**Objetivo:** Minimizar suma de variables artificiales")
                        else:
                            phase2_container.header("Fase 2: Optimizar Función Objetivo")
                            phase2_container.markdown("**Objetivo:** Optimizar función objetivo original")
                    
                    iteration_counters[phase] = iteration
                    
                    container = phase1_container if phase == 1 else phase2_container
                    
                    with container.expander(f"Iteración {iteration}", expanded=(iteration == 0)):
                        if iteration_info['entering'] is not None:
                            entering_idx = iteration_info['entering']
                            var_names = solver._get_variable_names()
                            entering_var = var_names[entering_idx] if entering_idx < len(var_names) else f"Col {entering_idx}"
                            
                            leaving_idx = iteration_info['leaving']
                            if leaving_idx is not None:
                                basic_vars = iteration_info['basic_vars']
                                if leaving_idx < len(basic_vars):
                                    leaving_var_idx = basic_vars[leaving_idx]
                                    leaving_var = var_names[leaving_var_idx] if leaving_var_idx < len(var_names) else f"Var {leaving_var_idx}"
                                else:
                                    leaving_var = f"Fila {leaving_idx}"
                            else:
                                leaving_var = "N/A"
                            
                            pivot_value = iteration_info['pivot']
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Variable Entrante", entering_var)
                            with col2:
                                st.metric("Variable Saliente", leaving_var)
                            with col3:
                                st.metric("Elemento Pivote", f"{pivot_value:.4f}")
                        
                        tableau = iteration_info['tableau']
                        basic_vars = iteration_info['basic_vars']
                        var_names = solver._get_variable_names()
                        
                        if phase == 2:
                            artificial_start = solver.n_vars + solver.n_slack + solver.n_surplus
                            display_var_names = var_names[:artificial_start]
                            if tableau.shape[1] > len(display_var_names) + 1:
                                tableau_display = tableau[:, :len(display_var_names) + 1]
                            else:
                                tableau_display = tableau
                        else:
                            display_var_names = var_names
                            tableau_display = tableau
                        
                        df = create_tableau_dataframe(
                            tableau_display, 
                            display_var_names, 
                            phase,
                            basic_vars if phase == 1 else None
                        )
                        
                        st.dataframe(df, use_container_width=True)
                
                final_result = {
                    'status': solver.status,
                    'solution': {},
                    'optimal_value': solver.optimal_value,
                    'message': ''
                }
                
                if solver.status == 'optimal':
                    final_result['solution'] = solver._extract_solution()
                    final_result['message'] = 'Solución óptima encontrada'
                elif solver.status == 'infeasible':
                    final_result['message'] = 'El problema es infactible'
                elif solver.status == 'unbounded':
                    final_result['message'] = 'El problema es no acotado'
                else:
                    final_result['message'] = 'Error en el proceso'
                    
            except StopIteration as e:
                if hasattr(e, 'value') and e.value:
                    final_result = e.value
                else:
                    final_result = {
                        'status': solver.status,
                        'solution': {},
                        'optimal_value': solver.optimal_value,
                        'message': 'Proceso completado'
                    }
            
            st.header("Resultado Final")
            
            if solver.status == 'optimal':
                st.success("Solución Óptima Encontrada")
                
                solution = final_result.get('solution', {})
                if not solution:
                    solution = solver._extract_solution()
                
                st.subheader("Valores de las Variables")
                solution_df = pd.DataFrame([
                    {'Variable': var, 'Valor': val}
                    for var, val in solution.items()
                    if abs(val) > 1e-10 or var.startswith('x')
                ])
                st.dataframe(solution_df, use_container_width=True)
                
                optimal_val = final_result.get('optimal_value') or solver.optimal_value
                if optimal_val is not None:
                    st.subheader("Valor Óptimo")
                    st.latex(f"Z^* = {optimal_val:.4f}")
                
            elif solver.status == 'infeasible':
                st.error("Problema Infactible")
                st.markdown("""
                No existen valores que satisfagan todas las restricciones simultáneamente.
                """)
                
            elif solver.status == 'unbounded':
                st.warning("Problema No Acotado")
                st.markdown("""
                La función objetivo puede crecer indefinidamente sin violar las restricciones.
                """)
                
            elif solver.status == 'degenerate':
                st.info("Solución Degenerada Detectada")
                st.markdown("""
                Variables artificiales en la base con valor cero. La solución puede no ser única.
                """)
                
                solution = solver._extract_solution()
                st.subheader("Valores de las Variables")
                solution_df = pd.DataFrame([
                    {'Variable': var, 'Valor': val}
                    for var, val in solution.items()
                    if abs(val) > 1e-10 or var.startswith('x')
                ])
                st.dataframe(solution_df, use_container_width=True)
                
                if solver.optimal_value is not None:
                    st.subheader("Valor Óptimo")
                    st.latex(f"Z^* = {solver.optimal_value:.4f}")
            else:
                st.error("Error en el proceso de solución")
                if final_result and final_result.get('message'):
                    st.error(final_result['message'])
        
        except ValueError as e:
            st.error(f"Error de validación: {e}")
        except Exception as e:
            st.error(f"Error inesperado: {e}")
            st.exception(e)
    
    else:
        st.info("Configura el problema e ingresa los coeficientes, luego haz clic en Resolver.")
        
        st.markdown("""
        ### Ejemplo de Uso
        
        **Problema:**
        - Maximizar: Z = 3x₁ + 2x₂
        - Sujeto a:
          - 2x₁ + x₂ ≤ 18
          - 2x₁ + 3x₂ ≥ 42
          - 3x₁ + x₂ = 24
        
        **Configuración:**
        1. Tipo: Maximizar
        2. Variables: 2
        3. Restricciones: 3
        4. Función Objetivo: 3, 2
        5. Restricción 1: 2, 1, ≤, 18
        6. Restricción 2: 2, 3, ≥, 42
        7. Restricción 3: 3, 1, =, 24
        """)


if __name__ == "__main__":
    main()
