import numpy as np
from totalpersistence import totalpersistence, kercoker_via_cone
from totalpersistence.utils import general_position_distance_matrix


def test_totalpersistence_basic():
    # Create simple test data
    X = np.array([[0, 0], [1, 0], [0, 1]])  # Triangle vertices
    Y = np.array([[0, 0], [2, 0], [0, 2]])  # Scaled triangle vertices
    f = np.array([0, 1, 2])  # Simple function values

    dX = general_position_distance_matrix(X)
    dY = general_position_distance_matrix(Y)

    coker_dgm, ker_dgm, cone_dgm, dgmX, dgmY = kercoker_via_cone(dX, dY, f, maxdim=2, cone_eps=0, tol=1e-11)

    # Call totalpersistence
    coker_bottleneck_distances, ker_bottleneck_distances, coker_matchings, ker_matchings = totalpersistence(
        coker_dgm, ker_dgm
    )

    # TODO: Add assertions for a relevant case

    # Basic assertions
    assert isinstance(coker_bottleneck_distances, list)
    assert isinstance(ker_bottleneck_distances, list)
    assert isinstance(coker_matchings, list)
    assert isinstance(ker_matchings, list)

    # The distance should be non-negative
    assert all(d >= 0 for d in coker_bottleneck_distances)
    assert all(d >= 0 for d in ker_bottleneck_distances)
