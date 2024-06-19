"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import torch


def geodesic(manifold, start_point, end_point):
    shooting_tangent_vec = manifold.logmap(start_point, end_point) # log maps compute a vector in the tangent space given starting and ending points
    def path(t):
        """Generate parameterized function for geodesic curve.
        Parameters
        ----------
        t : array-like, shape=[n_points,]
            Times at which to compute points of the geodesics.
        """
        tangent_vecs = torch.einsum("i,...k->...ik", t, shooting_tangent_vec) # this essentially compute scaled tangent vectors by time points t, this is because we are doing interpolation along manifolds, vectors don't change, but scaled by time
        points_at_time_t = manifold.expmap(start_point.unsqueeze(-2), tangent_vecs) # given starting points and vectors in the tangent space, exp maps compute where the points would land on the manifold
        return points_at_time_t

    return path # the return value is a function that given a set of time points return a path on the manifold
