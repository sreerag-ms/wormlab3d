from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


@torch.jit.script
def normalise(X: torch.Tensor) -> torch.Tensor:
    return X / X.norm(dim=-1, keepdim=True, p=2)


@torch.jit.script
def an_orthonormal(X: torch.Tensor) -> torch.Tensor:
    bs = X.shape[0]
    device = X.device
    Y = torch.stack([
        X[:, 1],
        -X[:, 0],
        torch.zeros(bs, device=device)
    ], dim=-1)
    Y = normalise(Y)
    return Y


@torch.jit.script
def orthogonalise(source: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return source - (torch.einsum('bs,bs->b', source, ref) / ref.norm(dim=-1, keepdim=False, p=2))[:, None] * ref


@torch.jit.script
def avg_pool_2d(x: torch.Tensor, oob_grad_val: float = 0., mode: str = 'constant') -> torch.Tensor:
    """
    Average pooling with overlap and boundary values.
    """
    padded_grad = F.pad(x, (1, 1, 1, 1), mode=mode, value=oob_grad_val)
    ag = F.avg_pool2d(padded_grad, kernel_size=3, stride=2, padding=0)
    return ag


@torch.jit.script
def make_gaussian_kernel(sigma: float, device: torch.device) -> torch.Tensor:
    ks = int(sigma * 5)
    if ks % 2 == 0:
        ks += 1
    ts = torch.linspace(-ks // 2, ks // 2 + 1, ks, device=device)
    gauss = torch.exp((-(ts / sigma)**2 / 2))
    kernel = gauss / gauss.sum()

    return kernel


@torch.jit.script
def smooth_parameter(param: torch.Tensor, ks: int, mode: str = 'avg') -> torch.Tensor:
    """
    Apply 1D average pooling to smooth a parameter vector.
    """
    pad_size = int(ks / 2)
    p_smoothed = torch.cat([
        torch.repeat_interleave(param[:, 0].unsqueeze(1), pad_size, dim=1),
        param,
        torch.repeat_interleave(param[:, -1].unsqueeze(1), pad_size, dim=1),
    ], dim=1)

    # Average pooling smoothing
    if mode == 'avg':
        p_smoothed = p_smoothed[:, None, :]
        p_smoothed = F.avg_pool1d(p_smoothed, kernel_size=ks, stride=1, padding=0)
        p_smoothed = p_smoothed.squeeze(1)

    # Smooth with a gaussian kernel
    elif mode == 'gaussian':
        k_sig = ks / 5
        if param.ndim == 2:
            param = param.unsqueeze(-1)
        n_channels = param.shape[-1]
        k = make_gaussian_kernel(k_sig, device=param.device)
        k = torch.stack([k] * n_channels)[:, None, :]
        p_smoothed = p_smoothed.permute(0, 2, 1)
        p_smoothed = F.conv1d(p_smoothed, weight=k, groups=n_channels)
        p_smoothed = p_smoothed.permute(0, 2, 1)

    else:
        raise RuntimeError(f'Unknown smoothing mode: "{mode}".')

    return p_smoothed


@torch.jit.script
def make_rotation_matrix(
        cos_phi: torch.Tensor,
        sin_phi: torch.Tensor,
        cos_theta: torch.Tensor,
        sin_theta: torch.Tensor,
        cos_psi: torch.Tensor,
        sin_psi: torch.Tensor
) -> torch.Tensor:
    """
    Build the full rotation matrix from the rotation preangles (sin(.), cos(.)).
    """
    return torch.stack([
        torch.stack([
            cos_theta * cos_phi,
            sin_psi * sin_theta * cos_phi - cos_psi * sin_phi,
            cos_psi * sin_theta * cos_phi + sin_psi * sin_phi
        ]),
        torch.stack([
            cos_theta * sin_phi,
            sin_psi * sin_theta * sin_phi + cos_psi * cos_phi,
            cos_psi * sin_theta * sin_phi - sin_psi * cos_phi
        ]),
        torch.stack([
            -sin_theta,
            sin_psi * cos_theta,
            cos_psi * cos_theta
        ])
    ])


@torch.jit.script
def generate_residual_targets(
        masks_target: List[torch.Tensor],
        masks: List[torch.Tensor],
        detection_masks: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    Generate the target masks including residual passed up from high-resolution to low-resolution.
    """
    D = len(masks_target)
    diff_eps = 0.01
    targets = []

    # Calculate the masks diffs
    masks_diffs = []
    for d in range(D):
        diff = torch.clamp(masks_target[d] - masks[d], min=0)
        diff[diff < diff_eps] = 0
        masks_diffs.append(diff)

    for d in range(D):
        target_d = masks_target[d]

        # Add residual - anything missed at lower depth is added on to the higher level target
        residual_next = torch.zeros_like(target_d)
        for d2 in range(d + 1, D):
            sf = 1 / (2**(d2 - d))
            residual_next += sf * masks_diffs[d2]
        target_d = target_d + residual_next

        # Only allow the target through where there was something detected at this depth or the next
        if d < D - 1:
            dm = torch.amax(torch.stack([detection_masks[d], detection_masks[d + 1]]), dim=0)
        elif 0 and d == D - 1:
            dm = torch.ones_like(detection_masks[d])  # Let everything through at the deepest level
        else:
            dm = detection_masks[d]
        target_d = target_d * dm
        target_d = torch.clamp(target_d, min=0, max=1)
        target_d = target_d.detach()
        targets.append(target_d)

    return targets


@torch.jit.script
def _calculate_derivative(
        points: torch.Tensor,
        spacings: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the derivative using finite differences.
    https://numpy.org/doc/stable/reference/generated/numpy.gradient.html
    """
    # Step sizes are non-homogeneous
    h = spacings[..., None]
    hs = h[:, :-1]
    hd = h[:, 1:]
    hs2 = hs**2
    hd2 = hd**2

    # Calculate derivative of interior points using central differences
    numerator = hs2 * points[:, 2:] \
                + (hd2 - hs2) * points[:, 1:-1] \
                - hd2 * points[:, :-2]
    denominator = hs * hd * (hd + hs)
    dp_int = numerator / denominator

    # Calculate boundary point derivatives using forward/backward differences
    dp_s = (points[:, 1] - points[:, 0]) / h[:, 0]
    dp_d = (points[:, -1] - points[:, -2]) / h[:, -1]

    # Combine to give derivative at every point
    derivative = torch.cat([dp_s[:, None, :], dp_int, dp_d[:, None, :]], dim=1)

    return derivative


@torch.jit.script
def calculate_curvature(
        points: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the vector curvature along the body.
    """
    N = points.shape[1]
    if N > 2:
        q = torch.norm(points[:, 1:] - points[:, :-1], dim=-1)
        T = _calculate_derivative(points, q)
        T_norm = torch.norm(T, dim=-1, keepdim=True)
        T = T / T_norm
        du = torch.ones_like(q) * q.sum(dim=-1, keepdim=True) / (N - 1)
        K = _calculate_derivative(T, du)
        K = K / T_norm
    else:
        K = torch.zeros_like(points, device=points.device)
    return K


@torch.jit.script
def _update_frame(
        m1: torch.Tensor,
        m2: torch.Tensor,
        M1: torch.Tensor,
        M2: torch.Tensor,
        T: torch.Tensor,
        h: torch.Tensor,
        idx: int,
        direction: int,
):
    """
    Update the orthonormal frame (T/M1/M2) from idx moving either forwards or backwards along the curve.
    """
    k1 = m1[:, idx][:, None]
    k2 = m2[:, idx][:, None]

    if direction == 1:
        idx_prev = idx - 1
        idx_next = idx
        ss = 1
    else:
        idx_prev = idx
        idx_next = idx - 1
        ss = -1

    dTds = k1 * M1[:, idx_prev].clone() + k2 * M2[:, idx_prev].clone()
    dM1ds = -k1 * T[:, idx_prev].clone()
    dM2ds = -k2 * T[:, idx_prev].clone()

    T_tilde = T[:, idx_prev].clone() + ss * h * dTds
    M1_tilde = M1[:, idx_prev].clone() + ss * h * dM1ds
    M2_tilde = M2[:, idx_prev].clone() + ss * h * dM2ds

    T[:, idx_next] = normalise(T_tilde)
    M1[:, idx_next] = normalise(M1_tilde)
    M2[:, idx_next] = normalise(M2_tilde)


@torch.jit.script
def integrate_curvature(
        X0: torch.Tensor,
        T0: torch.Tensor,
        l: torch.Tensor,
        K: torch.Tensor,
        M10: Optional[torch.Tensor] = None,
        start_idx: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Starting from vertex point X0 with tangent T0 and optional frame direction M10 at start_idx,
    integrate the curvature to produce a curve of length l.
    """
    bs = X0.shape[0]
    N = K.shape[1]
    if start_idx is None:
        start_idx = int((N - 1) / 2)
    else:
        assert 0 <= start_idx < N
    device = X0.device
    shape = (bs, N, 3)

    # Outputs
    X = torch.zeros(shape, device=device)
    T = torch.zeros(shape, device=device)
    M1 = torch.zeros(shape, device=device)
    M2 = torch.zeros(shape, device=device)

    # Step size to build a curve of length l
    h = (l / (N - 1))[:, None]

    # Curvature is in units assuming length 1
    # m1 = K[:, :, 0] / l[:, None]
    # m2 = K[:, :, 1] / l[:, None]
    m1 = K[:, :, 0] / h
    m2 = K[:, :, 1] / h

    # Initial frame values
    T0 = normalise(T0)
    T[:, start_idx] = T0
    if M10 is None:
        M10 = an_orthonormal(T0)
    else:
        M10 = orthogonalise(M10, T0)
        M10 = normalise(M10)
    M1[:, start_idx] = M10
    M2[:, start_idx] = torch.linalg.cross(T[:, start_idx].clone(), M1[:, start_idx].clone())

    # Calculate orthonormal frame from the middle-out
    for i in range(start_idx + 1, N):
        _update_frame(m1, m2, M1, M2, T, h, i, 1)
    for i in range(start_idx, 0, -1):
        _update_frame(m1, m2, M1, M2, T, h, i, -1)

    # Calculate curve coordinates
    X[:, start_idx] = X0
    for i in range(start_idx - 1, -1, -1):
        X[:, i] = X[:, i + 1] - h * T[:, i]
    for i in range(start_idx + 1, N):
        X[:, i] = X[:, i - 1] + h * T[:, i - 1]

    return X, T, M1


@torch.jit.script
def integrate_curvature_combined(
        X0: torch.Tensor,
        T0: torch.Tensor,
        l: torch.Tensor,
        K: torch.Tensor,
        M10: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Starting from midpoint X0 with tangent T0 and optional frame direction M10,
    integrate the curvature to produce a curve of length l.
    Same as above but update frame included in loops, kept for posterity.
    """
    bs = X0.shape[0]
    N = K.shape[1]
    N2 = int(N / 2)
    device = X0.device
    shape = (bs, N, 3)

    # Outputs
    X = torch.zeros(shape, device=device)
    T = torch.zeros(shape, device=device)
    M1 = torch.zeros(shape, device=device)
    M2 = torch.zeros(shape, device=device)

    # Step size to build a curve of length l
    h = (l / (N - 1))[:, None]

    # Curvature is in units assuming length 1
    # m1 = K[:, :, 0] / l[:, None]
    # m2 = K[:, :, 1] / l[:, None]
    m1 = K[:, :, 0] / h
    m2 = K[:, :, 1] / h

    # Initial frame values
    T0 = normalise(T0)
    T[:, N2 - 1] = T0
    if M10 is None:
        M10 = an_orthonormal(T0)
    else:
        M10 = orthogonalise(M10, T0)
        M10 = normalise(M10)
    M1[:, N2 - 1] = M10
    M2[:, N2 - 1] = torch.linalg.cross(T[:, N2 - 1].clone(), M1[:, N2 - 1].clone())

    # Calculate orthonormal frame from the middle-out
    for i in range(N2, N):
        k1 = m1[:, i][:, None]
        k2 = m2[:, i][:, None]

        dTds = k1 * M1[:, i - 1].clone() + k2 * M2[:, i - 1].clone()
        dM1ds = -k1 * T[:, i - 1].clone()
        dM2ds = -k2 * T[:, i - 1].clone()

        T_tilde = T[:, i - 1].clone() + h * dTds
        M1_tilde = M1[:, i - 1].clone() + h * dM1ds
        M2_tilde = M2[:, i - 1].clone() + h * dM2ds

        T[:, i] = normalise(T_tilde)
        M1[:, i] = normalise(M1_tilde)
        M2[:, i] = normalise(M2_tilde)

    for i in range(N2 - 1, 0, -1):
        k1 = m1[:, i][:, None]
        k2 = m2[:, i][:, None]

        dTds = k1 * M1[:, i].clone() + k2 * M2[:, i].clone()
        dM1ds = -k1 * T[:, i].clone()
        dM2ds = -k2 * T[:, i].clone()

        T_tilde = T[:, i].clone() - h * dTds
        M1_tilde = M1[:, i].clone() - h * dM1ds
        M2_tilde = M2[:, i].clone() - h * dM2ds

        T[:, i - 1] = normalise(T_tilde)
        M1[:, i - 1] = normalise(M1_tilde)
        M2[:, i - 1] = normalise(M2_tilde)

    # Calculate curve coordinates
    X[:, N2 - 1] = X0
    X[:, N2] = X0 + T0 * h
    for i in range(N2 - 2, -1, -1):
        X[:, i] = X[:, i + 1] - h * T[:, i]
    for i in range(N2 + 1, N):
        X[:, i] = X[:, i - 1] + h * T[:, i - 1]

    return X, T, M1


@torch.jit.script
def loss_(m: str, x: torch.Tensor, y: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    if m == 'mse':
        # l = F.mse_loss(x, y, reduction='mean')
        l = F.mse_loss(x, y, reduction='mean' if reduce else 'none')
    elif m == 'kl':
        l = F.kl_div(x, y, reduction='batchmean' if reduce else 'none')
    elif m == 'logdiff':
        l = torch.sum((torch.log(1 + x) - torch.log(1 + y))**2)
        # todo: is this worth it? if so needs parametrising
        # l = torch.sum(
        #     torch.where(
        #         x > y,
        #         torch.log(1 + 2 * x) - torch.log(1 + y),
        #         torch.log(1 + x) - torch.log(1 + y)
        #     )**2
        # )
    elif m == 'bce':
        l = F.binary_cross_entropy(x, y, reduction='mean' if reduce else 'none')
    else:
        l = torch.tensor(0., device=x.device)
    return l


@torch.jit.script
def calculate_renders_losses(
        masks: List[torch.Tensor],
        masks_target: List[torch.Tensor],
        metric: str,
        multiscale: bool = False
) -> List[torch.Tensor]:
    """
    Calculate the losses between the given masks and the targets.
    """
    D = len(masks)
    losses = []

    for d in range(D):
        masks_d = masks[d]
        target_d = masks_target[d]

        if multiscale:
            # Multiscale loss
            masks_rep = masks_d.clone()
            target_rep = target_d.clone()
            loss = torch.tensor(0., device=masks_rep.device)
            k = 1
            while masks_rep.shape[-1] > 1:
                loss += loss_(metric, masks_rep, target_rep)

                # Downsample using 3x3 average pooling with stride of 2
                masks_rep = avg_pool_2d(masks_rep, oob_grad_val=0.)
                target_rep = avg_pool_2d(target_rep, oob_grad_val=0.)
                k += 1
        else:
            loss = loss_(metric, masks_d, target_d)

        losses.append(loss)

    return losses


@torch.jit.script
def calculate_neighbours_losses(
        points: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    The distance between neighbours should be the same.
    """
    D = len(points)
    losses = []
    for d in range(D):
        points_d = points[d]

        # Take central differences
        if points_d.shape[1] > 2:
            dist_ltr = torch.norm(points_d[:, 1:-1] - points_d[:, :-2], dim=-1)
            dist_rtl = torch.norm(points_d[:, 2:] - points_d[:, 1:-1], dim=-1)
            loss = torch.sum((torch.log(1 + dist_ltr) - torch.log(1 + dist_rtl))**2)
        else:
            loss = torch.tensor(0., device=points[0].device)

        losses.append(loss)

    return losses


@torch.jit.script
def calculate_parents_losses(
        points: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    The distance between points and their parent should be equal siblings.
    """
    D = len(points)
    losses = [torch.tensor(0., device=points[0].device), ]
    for d in range(1, D):
        points_d = points[d]
        parents = points[d - 1].detach()
        points_parent = torch.repeat_interleave(parents, repeats=2, dim=1)

        # Calculate distances to parents
        dists = torch.norm(points_d - points_parent, dim=-1)

        # Distance to parent should be same for siblings
        # loss_equidistant = torch.sum((torch.log(1 + dists[:, ::2]) - torch.log(1 + dists[:, 1::2]))**2)

        # Distance from child to parent should be equal to half the distance of the parent to it's neighbour
        if parents.shape[1] > 2:
            left_dist_target = torch.norm(parents[:, 1:] - parents[:, :-1], dim=-1) / 4
            right_dist_target = torch.norm(parents[:, :-1] - parents[:, 1:], dim=-1) / 4
            left_children_to_parent = dists[:, ::2]
            right_children_to_parent = dists[:, 1::2]
            loss_left_children = torch.sum(
                (torch.log(1 + left_dist_target) -
                 torch.log(1 + left_children_to_parent[:, 1:]))**2
            )
            loss_right_children = torch.sum(
                (torch.log(1 + right_dist_target) -
                 torch.log(1 + right_children_to_parent[:, :-1]))**2
            )
            loss = loss_left_children + loss_right_children
        else:
            loss = torch.sum((torch.log(1 + dists[:, 0]) - torch.log(1 + dists[:, 1]))**2)

        losses.append(loss)

    return losses


@torch.jit.script
def calculate_parents_losses_curvatures(
        X0: List[torch.Tensor],
        T0: List[torch.Tensor],
        M10: List[torch.Tensor],
        X0ht: List[torch.Tensor],
        T0ht: List[torch.Tensor],
        M10ht: List[torch.Tensor],
        length: List[torch.Tensor],
        curvatures: List[torch.Tensor],
        curvatures_smoothed: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    The curvatures between child and parent should be close.
    """
    D = len(X0)
    losses = [torch.tensor(0., device=curvatures[0].device), ]
    for d in range(1, D):
        # X0 (midpoint) should be close between parent and child
        loss_X0 = torch.sum((X0[d] - X0[d - 1].detach())**2)

        # T0 (tangent at midpoint) direction should be similar
        loss_T0 = torch.sum((T0[d] - T0[d - 1].detach())**2)

        # M10 (frame at midpoint) direction should be similar
        loss_M10 = torch.sum((M10[d] - M10[d - 1].detach())**2)

        # X0ht (end points) should be close between parent and child
        loss_X0ht = torch.sum((X0ht[d] - X0ht[d - 1].detach())**2)

        # T0ht (tangents at end points) direction should be similar
        loss_T0ht = torch.sum((T0ht[d] - T0ht[d - 1].detach())**2)

        # M10 (frame at end points) direction should be similar
        loss_M10ht = torch.sum((M10ht[d] - M10ht[d - 1].detach())**2)

        # Lengths should be similar
        loss_l = torch.sum((length[d] - length[d - 1].detach())**2)

        # Curvature values should be close
        Ks_d = curvatures_smoothed[d]
        Ks_p = curvatures_smoothed[d - 1].detach()
        Ks_p = torch.repeat_interleave(Ks_p, repeats=2, dim=1)
        loss_K = torch.sum((Ks_d - Ks_p)**2)

        loss = loss_X0 + loss_T0 + loss_M10 + loss_X0ht + loss_T0ht + loss_M10ht + loss_l + loss_K
        losses.append(loss)

    return losses


@torch.jit.script
def calculate_aunts_losses(
        points: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    The distance between non-sibling neighbouring points should be equal to the
    sum of the distances between those points and their parents.
    """
    D = len(points)
    losses = [torch.tensor(0., device=points[0].device), ]
    for d in range(1, D):
        points_d = points[d]
        parents = points[d - 1]

        if parents.shape[1] > 2:
            left_children = points_d[:, ::2]
            right_children = points_d[:, 1::2]
            left_children_to_left_aunt = torch.norm(left_children[:, 1:] - parents[:, :-1], dim=-1)**2
            right_children_to_right_aunt = torch.norm(right_children[:, :-1] - parents[:, 1:], dim=-1)**2
            left_children_to_right_children = torch.norm(left_children[:, 1:] - right_children[:, :-1], dim=-1)**2

            a = left_children_to_right_children
            b = (left_children_to_left_aunt + right_children_to_right_aunt)
            loss = torch.sum((torch.log(1 + a) - torch.log(1 + b))**2)
        else:
            loss = torch.tensor(0., device=points[0].device)
        losses.append(loss)

    return losses


@torch.jit.script
def calculate_scores_losses(
        scores: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    The scores should be even along the body.
    """
    D = len(scores)
    losses = []
    for d in range(D):
        scores_d = scores[d]

        # # Scores should be even along body
        # loss = torch.sum(
        #     (torch.log(1 + scores_d)
        #      - torch.log(1 + scores_d.mean(dim=1, keepdim=True).detach()))**2
        # )

        # Weight scores with a quadratic with y=0 at x=0.5 and y=ht_weight at x=[0,1]
        ht_weight = 1
        sf = 4 * ht_weight * (torch.linspace(0, 1, scores_d.shape[1], device=scores_d.device) - 0.5)**2
        scaled_scores = scores_d * sf[None, ...]

        # Scores should be maximised
        loss = torch.sum(torch.amax(scores_d, dim=-1).detach() / (torch.mean(scaled_scores, dim=-1) + 1e-8))

        # Symmetry loss
        N2 = int(scores_d.shape[1] / 2)
        loss_sym = torch.sum(
            ((scores_d[:, :N2].flip(dims=(1,)) - scores_d[:, N2:]) / torch.amax(scores_d, dim=-1, keepdim=True))**2
        )
        loss = loss + loss_sym

        losses.append(loss)

    return losses


@torch.jit.script
def calculate_smoothness_losses(
        points: List[torch.Tensor],
        points_smoothed: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    The points should change smoothly along the body.
    """
    D = len(points)
    losses = []
    for d in range(D):
        points_d = points[d]
        if points_d.shape[1] > 4:
            points_smoothed_d = points_smoothed[d]
            loss = torch.sum((points_d - points_smoothed_d)**2)
        else:
            loss = torch.tensor(0., device=points[0].device)
        losses.append(loss)

    return losses


@torch.jit.script
def calculate_smoothness_losses_curvatures(
        curvatures: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    The curvatures should change smoothly along the body.
    """
    D = len(curvatures)
    losses = []
    for d in range(D):
        K_d = curvatures[d]
        if K_d.shape[1] > 4:
            ds = ((K_d[:, 1:] - K_d[:, :-1])**2).sum(dim=(-1, -2)).mean()
            ds2 = ((K_d[:, 2:] - K_d[:, :-2])**2).sum(dim=(-1, -2)).mean() / 2
            loss = ds + ds2
        else:
            loss = torch.tensor(0., device=curvatures[0].device)
        losses.append(loss)

    return losses


@torch.jit.script
def calculate_curvature_losses(
        points: List[torch.Tensor],
        curvatures: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    The curvatures should not be too large.
    """
    D = len(points)
    device = points[0].device
    losses = []
    for d in range(D):
        points_d = points[d]

        if points_d.shape[1] > 2:
            k = torch.norm(curvatures[d], dim=-1)
            loss = (k**2).sum()

        # if points_d.shape[1] > 4:
        #     sl = torch.norm(points_d[:, 1:] - points_d[:, :-1], dim=-1)
        #     wl = sl.sum(dim=-1, keepdim=True)
        #
        #     # Only penalise curvatures greater than 2-revolutions
        #     kinks = k > (2 * 2 * torch.pi) / wl
        #     loss = k[kinks].sum()
        else:
            loss = torch.tensor(0., device=device)
        losses.append(loss)

    return losses


@torch.jit.script
def calculate_curvature_losses_curvatures(
        curvatures: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    The curvatures should not be too large.
    """
    D = len(curvatures)
    losses = []
    for d in range(D):
        loss = (curvatures[d]**2).sum()
        losses.append(loss)

    return losses


@torch.jit.script
def calculate_curvature_losses_curvature_deltas(
        curvatures: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    The curvatures should not be too large.
    """
    D = len(curvatures)
    losses = []
    for d in range(D):
        K_d = curvatures[d]

        # Regularise the curvature of main curve
        loss0 = (K_d[0]**2).sum()

        # Regularise the deltas
        if K_d.shape[0] > 1:
            dKs = K_d[1:]
            loss_dK = (dKs**2).sum()
        else:
            loss_dK = torch.tensor(0., device=K_d.device)

        losses.append(loss0 + loss_dK)

    return losses


@torch.jit.script
def calculate_temporal_losses(
        points: List[torch.Tensor],
        points_prev: Optional[List[torch.Tensor]],
) -> List[torch.Tensor]:
    """
    The points should change smoothly in time.
    """
    D = len(points)

    # If there are no other time points available then just return zeros.
    if points_prev is None and points[0].shape[0] == 1:
        return [torch.tensor(0., device=points[0].device) for _ in range(D)]

    losses = []
    for d in range(D):
        points_d = points[d]

        # Prepend the previous points
        if points_prev is not None:
            points_prev_d = points_prev[d].unsqueeze(0)
            points_d = torch.cat([points_prev_d, points_d], dim=0)

        # Calculate losses to temporal neighbours
        loss_points = torch.sum((points_d[1:] - points_d[:-1])**2)

        # Lengths should not change much through time
        sl = torch.norm(points_d[:, 1:] - points_d[:, :-1], dim=-1)
        wl = sl.sum(dim=-1)
        loss_lengths = torch.sum(
            (torch.log(1 + wl)
             - torch.log(1 + wl.mean().detach()))**2
        )

        loss = loss_points + loss_lengths

        losses.append(loss)

    return losses


@torch.jit.script
def calculate_temporal_losses_curvatures(
        length: List[torch.Tensor],
        curvatures: List[torch.Tensor],
        X0: Optional[List[torch.Tensor]],
        T0: Optional[List[torch.Tensor]],
        M10: Optional[List[torch.Tensor]],
        X0ht: Optional[List[torch.Tensor]],
        T0ht: Optional[List[torch.Tensor]],
        M10ht: Optional[List[torch.Tensor]],
        length_prev: Optional[List[torch.Tensor]],
        curvatures_prev: Optional[List[torch.Tensor]],
        X0_prev: Optional[List[torch.Tensor]],
        T0_prev: Optional[List[torch.Tensor]],
        M10_prev: Optional[List[torch.Tensor]],
        X0ht_prev: Optional[List[torch.Tensor]],
        T0ht_prev: Optional[List[torch.Tensor]],
        M10ht_prev: Optional[List[torch.Tensor]],
) -> List[torch.Tensor]:
    """
    The curvatures should change smoothly in time.
    """
    D = len(curvatures)
    device = length[0].device

    # If there are no other time points available then just return zeros.
    if length_prev is None and len(length[0]) == 1:
        return [torch.tensor(0., device=device) for _ in range(D)]

    losses = []
    for d in range(D):
        # X0 (midpoint) should be close
        if X0 is not None:
            X0_d = X0[d]
            if X0_prev is not None:
                X0_prev_d = X0_prev[d].unsqueeze(0).detach()
                X0_d = torch.cat([X0_prev_d, X0_d], dim=0)
            loss_X0 = torch.sum((X0_d[1:] - X0_d[:-1])**2)
        else:
            loss_X0 = torch.tensor(0., device=device)

        # T0 (tangent at midpoint) direction should be similar
        if T0 is not None:
            T0_d = T0[d]
            if T0_prev is not None:
                T0_prev_d = T0_prev[d].unsqueeze(0).detach()
                T0_d = torch.cat([T0_prev_d, T0_d], dim=0)
            loss_T0 = torch.sum((T0_d[1:] - T0_d[:-1])**2)
        else:
            loss_T0 = torch.tensor(0., device=device)

        # M10 (frame at midpoint) direction should be similar
        if M10 is not None:
            M10_d = M10[d]
            if M10_prev is not None:
                M10_prev_d = M10_prev[d].unsqueeze(0).detach()
                M10_d = torch.cat([M10_prev_d, M10_d], dim=0)
            loss_M10 = torch.sum((M10_d[1:] - M10_d[:-1])**2)
        else:
            loss_M10 = torch.tensor(0., device=device)

        # X0ht (end points) should be close
        if X0ht is not None:
            X0ht_d = X0ht[d]
            if X0ht_prev is not None:
                X0ht_prev_d = X0ht_prev[d].unsqueeze(0).detach()
                X0ht_d = torch.cat([X0ht_prev_d, X0ht_d], dim=0)
            loss_X0ht = torch.sum((X0ht_d[1:] - X0ht_d[:-1])**2)
        else:
            loss_X0ht = torch.tensor(0., device=device)

        # T0ht (tangent at end points) direction should be similar
        if T0ht is not None:
            T0ht_d = T0ht[d]
            if T0ht_prev is not None:
                T0ht_prev_d = T0ht_prev[d].unsqueeze(0).detach()
                T0ht_d = torch.cat([T0ht_prev_d, T0ht_d], dim=0)
            loss_T0ht = torch.sum((T0ht_d[1:] - T0ht_d[:-1])**2)
        else:
            loss_T0ht = torch.tensor(0., device=device)

        # M10ht (frame at end points) direction should be similar
        if M10ht is not None:
            M10ht_d = M10ht[d]
            if M10ht_prev is not None:
                M10ht_prev_d = M10ht_prev[d].unsqueeze(0).detach()
                M10ht_d = torch.cat([M10ht_prev_d, M10ht_d], dim=0)
            loss_M10ht = torch.sum((M10ht_d[1:] - M10ht_d[:-1])**2)
        else:
            loss_M10ht = torch.tensor(0., device=device)

        # Lengths should be similar
        length_d = length[d]
        if length_prev is not None:
            length_prev_d = length_prev[d].unsqueeze(0).detach()
            length_d = torch.cat([length_prev_d, length_d], dim=0)
        loss_l = torch.sum((length_d[1:] - length_d[:-1])**2)

        # Curvature values should be close
        curvatures_d = curvatures[d]
        if curvatures_prev is not None:
            curvatures_prev_d = curvatures_prev[d].unsqueeze(0).detach()
            curvatures_d = torch.cat([curvatures_prev_d, curvatures_d], dim=0)
        loss_K = torch.sum((curvatures_d[1:] - curvatures_d[:-1])**2)

        loss = loss_X0 + loss_T0 + loss_M10 + loss_X0ht + loss_T0ht + loss_M10ht + loss_l + loss_K
        losses.append(loss)

    return losses


@torch.jit.script
def calculate_temporal_losses_curvature_deltas(
        X0: List[torch.Tensor],
        T0: List[torch.Tensor],
        M10: List[torch.Tensor],
        length: List[torch.Tensor],
        curvatures: List[torch.Tensor],
        X0_prev: Optional[List[torch.Tensor]],
        T0_prev: Optional[List[torch.Tensor]],
        M10_prev: Optional[List[torch.Tensor]],
        length_prev: Optional[List[torch.Tensor]],
        curvatures_prev: Optional[List[torch.Tensor]],
) -> List[torch.Tensor]:
    """
    The curve should change smoothly in time.
    """
    D = len(X0)

    # Calculate the temporal losses as usual between previous curve and main curve.
    losses = calculate_temporal_losses_curvatures(
        [length[d][0].unsqueeze(0) for d in range(D)],
        [curvatures[d][0].unsqueeze(0) for d in range(D)],
        [X0[d][0].unsqueeze(0) for d in range(D)],
        [T0[d][0].unsqueeze(0) for d in range(D)],
        [M10[d][0].unsqueeze(0) for d in range(D)],
        None,
        None,
        None,
        length_prev,
        curvatures_prev,
        X0_prev,
        T0_prev,
        M10_prev,
        None,
        None,
        None,
    )

    # If there are no deltas available then just return zeros.
    if len(X0[0]) == 1:
        return losses

    for d in range(D):
        # dX0 (midpoint) changes
        dX0 = X0[d][1:]
        loss_dX0 = torch.sum(dX0**2)

        # dT0 (initial tangent) changes
        dT0 = T0[d][1:]
        loss_dT0 = torch.sum(dT0**2)

        # dM10 (initial frame) changes
        dM10 = M10[d][1:]
        loss_dM10 = torch.sum(dM10**2)

        # dl (length) changes
        dl = length[d][1:]
        loss_dl = torch.sum(dl**2)

        # Curvature deltas
        dK = curvatures[d][1:]
        loss_dK = torch.sum(dK**2)

        loss = loss_dX0 + loss_dT0 + loss_dM10 + loss_dl + loss_dK
        losses.append(loss)

    return losses


@torch.jit.script
def calculate_intersection_losses_curvatures(
        points: List[torch.Tensor],
        sigmas_smoothed: List[torch.Tensor],
        max_curvature: float,
) -> List[torch.Tensor]:
    """
    The curve should not intersect itself.
    """
    D = len(points)
    bs = points[0].shape[0]
    device = points[0].device
    losses = []
    eps = 1e-8

    for d in range(D):
        loss_d = torch.tensor(0., device=device)
        points_d = points[d]
        sigmas_d = sigmas_smoothed[d]
        N = points_d.shape[1]

        for bi in range(bs):
            sigs = sigmas_d[bi]

            # Calculate distances between points
            dists = F.pdist(points_d[bi])

            # Check distances between sufficiently far apart points on the body
            pos_offset = int(N / max_curvature)
            for i in range(N - 1):
                if i + pos_offset > N:
                    break
                for j in range(i + pos_offset, N):
                    idx = N * i - i * (i + 1) // 2 + j - 1 - i
                    dij = dists[idx]
                    dist_threshold = sigs[i] + sigs[j]
                    if dij < dist_threshold:
                        loss_d += dist_threshold / (dij + eps)

        losses.append(loss_d)

    return losses


@torch.jit.script
def calculate_alignment_losses_curvatures(
        curvatures: List[torch.Tensor],
        curvatures_prev: Optional[List[torch.Tensor]],
) -> List[torch.Tensor]:
    """
    Consecutive curves should be aligned in shape space.
    """
    D = len(curvatures)
    bs = curvatures[0].shape[0]
    device = curvatures[0].device

    if curvatures_prev is None and bs == 1:
        return [torch.tensor(0., device=device) for _ in range(D)]

    losses = []
    for d in range(D):
        curvatures_d = curvatures[d]

        # Prepend previous curvatures to the batch
        if curvatures_prev is not None:
            curvatures_prev_d = curvatures_prev[d].unsqueeze(0).detach()
            curvatures_d = torch.cat([curvatures_prev_d, curvatures_d], dim=0)

        # Calculate optimal alignment rotation angle between frames
        mc_prev = torch.view_as_complex(torch.stack([curvatures_d[:-1, :, 0], curvatures_d[:-1, :, 1]], dim=-1))
        mc_next = torch.view_as_complex(torch.stack([curvatures_d[1:, :, 0], curvatures_d[1:, :, 1]], dim=-1))

        opt_angles = -torch.angle(torch.einsum('ij,ij->i', mc_prev, mc_next.conj()))
        loss_d = torch.mean(opt_angles.abs())
        losses.append(loss_d)

    return losses


@torch.jit.script
def calculate_consistency_losses_curvatures(
        X: List[torch.Tensor],
        T: List[torch.Tensor],
        M1: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    Curves should be consistent with integration from head, tail and midpoint.
    """
    D = len(X)
    losses = []

    for d in range(D):
        X_d = X[d]
        T_d = T[d]
        M1_d = M1[d]

        LX = X_d.var(dim=1).sum(dim=(-1, -2)).mean()
        LT = T_d.var(dim=1).sum(dim=(-1, -2)).mean()
        LM1 = M1_d.var(dim=1).sum(dim=(-1, -2)).mean()

        L = LX + LT + LM1
        losses.append(L)

    return losses


@torch.jit.script
def calculate_consistency_losses_curvatures_ht(
        X: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    Curves should be consistent with integration from head or tail.
    """
    D = len(X)
    losses = []

    for d in range(D):
        X_d = X[d]
        L_ht = (X_d[:, 1] - X_d[:, 2]).norm(dim=-1, p=2).sum(dim=-1).mean()
        losses.append(L_ht)

    return losses
