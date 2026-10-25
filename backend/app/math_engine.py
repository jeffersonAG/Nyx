"""Cálculo seguro de álgebra lineal: datos numéricos, nunca eval de expresiones."""
from __future__ import annotations

import math
from typing import Any

from sympy import Matrix, Rational


def number(value: Any):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Solo se permiten números finitos en las matrices.")
    if not math.isfinite(value) or abs(value) > 1_000_000:
        raise ValueError("Valor numérico fuera de rango (máximo 1.000.000).")
    # Racionales exactos para entradas decimales sin ejecutar código del modelo.
    return Rational(str(value))


def checked_matrix(raw: Any) -> Matrix:
    if not isinstance(raw, list) or not 1 <= len(raw) <= 8:
        raise ValueError("La matriz debe tener entre 1 y 8 filas.")
    if not isinstance(raw[0], list) or not 1 <= len(raw[0]) <= 8:
        raise ValueError("Las filas deben contener entre 1 y 8 valores.")
    width = len(raw[0])
    if any(not isinstance(row, list) or len(row) != width for row in raw):
        raise ValueError("Todas las filas de la matriz deben tener igual longitud.")
    return Matrix([[number(value) for value in row] for row in raw])


def checked_vector(raw: Any, limit: int = 8) -> Matrix:
    if not isinstance(raw, list) or not 1 <= len(raw) <= limit:
        raise ValueError("El vector debe contener entre 1 y 8 números.")
    return Matrix([number(value) for value in raw])


def encode(value: Any) -> Any:
    if isinstance(value, Matrix):
        return [[str(item) for item in value.row(i)] for i in range(value.rows)]
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    return str(value)


def matrix_operation(operation: str, a: Any,
                     b: Any = None, vector_b: Any = None) -> dict:
    if operation not in {
        "determinant", "rank", "transpose", "inverse", "multiply",
        "solve", "dot", "cross"
    }:
        raise ValueError("Operación matemática no reconocida.")
    matrix_a = checked_matrix(a)
    outcome: Any
    if operation == "rank":
        outcome = matrix_a.rank()
    elif operation == "transpose":
        outcome = matrix_a.T
    elif operation in {"determinant", "inverse"}:
        if matrix_a.rows != matrix_a.cols:
            raise ValueError("La matriz debe ser cuadrada.")
        if operation == "determinant":
            outcome = matrix_a.det()
        else:
            if matrix_a.det() == 0:
                raise ValueError("La matriz no es invertible (determinante cero).")
            outcome = matrix_a.inv()
    elif operation == "multiply":
        matrix_b = checked_matrix(b)
        if matrix_a.cols != matrix_b.rows:
            raise ValueError("Dimensiones incompatibles para multiplicación matricial.")
        outcome = matrix_a * matrix_b
    elif operation == "solve":
        rhs = checked_vector(vector_b)
        if matrix_a.rows != len(rhs):
            raise ValueError("Ax=b requiere un valor de b por cada fila de A.")
        solution = matrix_a.gauss_jordan_solve(rhs)
        outcome = solution[0]
        # Puede quedar parametrizado si hay infinitas soluciones.
        if solution[1].rows:
            raise ValueError("Sistema indeterminado; no hay solución única.")
    elif operation in ("dot", "cross"):
        # A debe ser fila o columna para evitar el producto de matrices por error.
        if matrix_a.rows != 1 and matrix_a.cols != 1:
            raise ValueError("Para dot/cross utiliza una matriz de una fila o columna.")
        v1 = list(matrix_a)
        v2_matrix = checked_matrix(b)
        if v2_matrix.rows != 1 and v2_matrix.cols != 1:
            raise ValueError("El segundo argumento debe ser un vector.")
        v2 = list(v2_matrix)
        if len(v1) != len(v2):
            raise ValueError("Los vectores deben tener la misma longitud.")
        if operation == "dot":
            outcome = sum(x * y for x, y in zip(v1, v2))
        else:
            if len(v1) != 3:
                raise ValueError("El producto cruz requiere dos vectores de 3 dimensiones.")
            outcome = Matrix(v1).cross(Matrix(v2))
    return {"operation": operation, "value": encode(outcome),
            "method": "SymPy local (aritmética racional exacta)"}
