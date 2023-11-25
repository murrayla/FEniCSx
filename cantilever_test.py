# FEniCSx Cantilever Test

# +==+===+==+==+
# Linear elasticity
# Left hand of beam will be clamped with homgenous Dirchlet boundary conditions
# Right hand will be under gravity 
# Fenics requires a weak form
# +==+===+==+==+

import dolfinx as fe

CANTILEVER_LENGTH = 1.0
CANTILEVER_WIDTH = 0.2

N_POINTS_LENGTH = 10
N_POINTS_WIDTH = 3

LAME_MU = 1.0
LAME_LAMBDA = 1.25
DENSITY = 1.0
ACCELERATION_DUE_TO_GRAVITY = 0.016

def main():
    mesh = fe.BoxMesh(
        fe.Point(0.0, 0.0, 0.0),
        fe.Point(CANTILEVER_LENGTH, CANTILEVER_WIDTH, CANTILEVER_WIDTH)
        N_POINTS_LENGTH,
        N_POINTS_WIDTH,
        N_POINTS_WIDTH
    )
    lagrange_vector_space_first_order = fe.VectorFunctionSpace(
        mesh, 
        "Lagrange",
        1,
    )

    # BCs
    def clamped_boundary(x, on_boundary):
        return on_boundary and x[0] < fe.DOLPHIN_EPS
    
    dircichlet_clamped_boundary = fe.DirichletBC(
        lagrange_vector_space_first_order,
        fe.Constant((0.0, 0.0, 0.0)),

    )

if __name__ == "__main__":
    main()