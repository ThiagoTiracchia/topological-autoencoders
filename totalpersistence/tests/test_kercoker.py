import numpy as np
from totalpersistence.totalpersistence import kercoker_via_cone, general_position_distance_matrix


def test_kercoker_via_cone():
    # Test input data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]])
    Y = np.array([[0, 2], [1, 2], [2, 2]])
    f = np.array([0, 0, 1, 1, 2, 2])

    # Compute perturbed distance matrices
    dX = general_position_distance_matrix(X, perturb=1e-13)
    dY = general_position_distance_matrix(Y, perturb=1e-13)

    # Run the function
    coker_dgm, ker_dgm, cone_dgm, dgmX, dgmY = kercoker_via_cone(dX, dY, f, maxdim=2, cone_eps=0)

    # Expected output
    # for coker in [array([[ 0., inf]])]
    # for ker in [array([[ 0.5,  1. ], [ 0.5,  1. ], [ 0. ,  1. ], [ 0. ,  1. ], [ 0. ,  1. ]),
    # array([[ 1. ,  1.41421356], [ 1. ,  1.41421356]])]

    expected_coker = np.array([[0.0, float("inf")]])
    expected_ker = [
        np.array([[0.5, 1.0], [0.5, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]),
        np.array([[1.0, 1.41421356], [1.0, 1.41421356]]),
    ]
    tol = 1e-10

    # Compare results
    assert len(coker_dgm) == 1
    assert np.allclose(coker_dgm[0], expected_coker, atol=tol)
    assert len(ker_dgm) == 2
    for r, e in zip(ker_dgm, expected_ker):
        assert np.allclose(r, e, atol=tol)
