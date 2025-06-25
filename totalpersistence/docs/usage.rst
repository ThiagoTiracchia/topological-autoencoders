Total Persistence Usage Guide
===========================

This guide demonstrates how to use the main functions in the totalpersistence package.

Basic Usage
----------

The package provides functions to compute total persistence diagrams using bottleneck distances. Here's a basic example:

.. code-block:: python

    import numpy as np
    from totalpersistence import totalpersistence, kercoker_via_cone
    from totalpersistence.utils import general_position_distance_matrix

    # Create sample point clouds
    X = np.array([[0, 0], [1, 0], [0, 1]])  # Triangle vertices
    Y = np.array([[0, 0], [2, 0], [0, 2]])  # Scaled triangle vertices
    f = np.array([0, 1, 2])  # Function values

    # Generate distance matrices
    dX = general_position_distance_matrix(X)
    dY = general_position_distance_matrix(Y)

    # Compute persistence diagrams
    coker_dgm, ker_dgm, cone_dgm, dgmX, dgmY = kercoker_via_cone(
        dX, dY, f, maxdim=2, cone_eps=0, tol=1e-11
    )

    # Calculate total persistence
    coker_distances, ker_distances, coker_matchings, ker_matchings = totalpersistence(
        coker_dgm, ker_dgm
    )

Key Functions
------------

kercoker_via_cone
~~~~~~~~~~~~~~~~

The ``kercoker_via_cone`` function computes persistence diagrams using the cone algorithm:

.. code-block:: python

    def kercoker_via_cone(dX, dY, f, maxdim=1, cone_eps=0, tol=1e-11):
        """
        Compute persistence diagrams using the cone algorithm.

        Parameters
        ----------
        dX : np.array
            Distance matrix of the source space in condensed form
        dY : np.array
            Distance matrix of the target space in condensed form
        f : np.array
            Function values
        maxdim : int, optional
            Maximum dimension to compute (default=1)
        cone_eps : float, optional
            Cone parameter (default=0)
        tol : float, optional
            Tolerance for numerical computations (default=1e-11)

        Returns
        -------
        tuple
            (coker_dgm, ker_dgm, cone_dgm, dgmX, dgmY)
        """

totalpersistence
~~~~~~~~~~~~~~~

The ``totalpersistence`` function computes bottleneck distances:

.. code-block:: python

    def totalpersistence(coker_dgm, ker_dgm):
        """
        Compute total persistence using bottleneck distances.

        Parameters
        ----------
        coker_dgm : list
            List of cokernel persistence diagrams
        ker_dgm : list
            List of kernel persistence diagrams

        Returns
        -------
        tuple
            (coker_bottleneck_distances, ker_bottleneck_distances,
             coker_matchings, ker_matchings)
        """

Utility Functions
---------------

The package includes several utility functions in ``utils.py``:

- ``general_position_distance_matrix(X, perturb=1e-7)``: Generate a distance matrix with small perturbations
- ``lipschitz(dX, dY)``: Compute the Lipschitz constant
- ``conematrix(DX, DY, DY_fy, eps)``: Create the cone matrix for persistence calculations

Example with Real Data
--------------------

Here's a complete example analyzing point cloud data:

.. code-block:: python

    import numpy as np
    from totalpersistence import totalpersistence, kercoker_via_cone
    from totalpersistence.utils import general_position_distance_matrix

    # Generate sample point clouds
    n_points = 10
    X = np.random.rand(n_points, 2)  # Source space points
    Y = 2 * np.random.rand(n_points, 2)  # Target space points
    f = np.arange(n_points)  # Function values

    # Compute distance matrices
    dX = general_position_distance_matrix(X)
    dY = general_position_distance_matrix(Y)

    # Calculate persistence diagrams
    coker_dgm, ker_dgm, cone_dgm, dgmX, dgmY = kercoker_via_cone(
        dX, dY, f, maxdim=2
    )

    # Compute total persistence
    results = totalpersistence(coker_dgm, ker_dgm)
    coker_distances, ker_distances, coker_matchings, ker_matchings = results

    # Print results
    print("Cokernel distances:", coker_distances)
    print("Kernel distances:", ker_distances)