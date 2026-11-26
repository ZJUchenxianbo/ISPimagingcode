"""Shared numerical function extracted from the supplied solver."""
import numpy as np
from numpy.polynomial import Legendre, Polynomial

def lagrange_basis(degree, points):
    interior = Legendre.basis(degree).deriv().roots()
    nodes = np.r_[-1.0, interior, 1.0]
    polynomials = [Polynomial.fromroots(np.delete(nodes, k)) for k in range(degree + 1)]
    polynomials = [poly / poly(node) for poly, node in zip(polynomials, nodes)]
    values = np.column_stack([poly(points) for poly in polynomials])
    derivatives = np.column_stack([poly.deriv()(points) for poly in polynomials])
    return values, derivatives
