from typing import List, Optional

import torch
import torch.nn.functional as F


@torch.jit.script
def avg_pool_2d(x: torch.Tensor, oob_grad_val: float = 0., mode: str = 'constant') -> torch.Tensor:
    """
    Average pooling with overlap and boundary values.
    """
    padded_grad = F.pad(x, (1, 1, 1, 1), mode=mode, value=oob_grad_val)
    ag = F.avg_pool2d(padded_grad, kernel_size=3, stride=2, padding=0)
    return ag


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

        # Only allow the target through where there was something detected
        target_d[detection_masks[d] < 0.01] = 0

        target_d = torch.clamp(target_d, min=0, max=1)
        target_d = target_d.detach()
        targets.append(target_d)

    return targets


@torch.jit.script
def loss_(m: str, x: torch.Tensor, y: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    if m == 'mse':
        # l = F.mse_loss(x, y, reduction='mean')
        l = F.mse_loss(x, y, reduction='mean' if reduce else 'none')
    elif m == 'kl':
        l = F.kl_div(x, y, reduction='batchmean' if reduce else 'none')
    elif m == 'logdiff':
        l = torch.sum((torch.log(1 + x) - torch.log(1 + y))**2)
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
    losses = [torch.tensor(0., device=points[0].device), ]
    for d in range(1, D):
        points_d = points[d]

        # Take central differences
        dist_ltr = torch.norm(points_d[:, 1:-1] - points_d[:, :-2], dim=-1)
        dist_rtl = torch.norm(points_d[:, 2:] - points_d[:, 1:-1], dim=-1)
        loss = torch.sum((torch.log(1 + dist_ltr) - torch.log(1 + dist_rtl))**2)
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

        left_children = points_d[:, ::2]
        right_children = points_d[:, 1::2]
        left_children_to_left_aunt = torch.norm(left_children[:, 1:] - parents[:, :-1], dim=-1)**2
        right_children_to_right_aunt = torch.norm(right_children[:, :-1] - parents[:, 1:], dim=-1)**2
        left_children_to_right_children = torch.norm(left_children[:, 1:] - right_children[:, :-1], dim=-1)**2

        a = left_children_to_right_children
        b = (left_children_to_left_aunt + right_children_to_right_aunt)
        loss = torch.sum((torch.log(1 + a) - torch.log(1 + b))**2)
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
        loss = torch.sum((torch.log(1 + scores_d) - torch.log(1 + scores_d.mean().detach()))**2)
        losses.append(loss)

    return losses


@torch.jit.script
def calculate_sigmas_losses(
        sigmas: List[torch.Tensor],
        sigmas_smoothed: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    The sigmas should be smooth along the body and not vary too much.
    """
    D = len(sigmas)
    losses = []
    for d in range(D):
        sigmas_d = sigmas[d]
        # sigmas_smoothed_d = sigmas_smoothed[d]

        # Smoothness loss
        # loss = torch.sum((sigmas_d - sigmas_smoothed_d)**2)

        # Penalise too much deviation from the mean
        # loss += 0.1 * torch.sum((torch.log(1 + sigmas_d) - torch.log(1 + sigmas_d.mean().detach()))**2)

        # Sigmas should be equal in the middle section but taper towards the ends
        if d > 1:
            n = sigmas_d.shape[1]
            mp = int(n / 2)
            sd1 = torch.clamp(sigmas_d[:, :mp - 1] - sigmas_d[:, 1:mp], min=0).sum()
            sd2 = torch.clamp(sigmas_d[:, mp + 1:] - sigmas_d[:, mp:-1], min=0).sum()
            qp = int(n / 4)
            middle_section = sigmas_d[:, qp:3 * qp]
            sd3 = torch.sum((torch.log(1 + middle_section) - torch.log(1 + middle_section.mean().detach()))**2)
            loss = 0.1 * sd1 + 0.1 * sd2 + sd3
        else:
            loss = torch.sum((torch.log(1 + sigmas_d) - torch.log(1 + sigmas_d.mean().detach()))**2)

        losses.append(loss)

    return losses


@torch.jit.script
def calculate_exponents_losses(
        exponents: List[torch.Tensor],
        exponents_smoothed: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    The exponents should be smooth along the body and not vary too much.
    """
    D = len(exponents)
    losses = []
    for d in range(D):
        exponents_d = exponents[d]
        exponents_smoothed_d = exponents_smoothed[d]

        # Smoothness loss
        loss = torch.sum((exponents_d - exponents_smoothed_d)**2)

        losses.append(loss)

    return losses


@torch.jit.script
def calculate_intensities_losses(
        intensities: List[torch.Tensor],
        intensities_smoothed: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    The intensities should be smooth along the body and not vary too much.
    """
    D = len(intensities)
    losses = []
    for d in range(D):
        intensities_d = intensities[d]
        intensities_smoothed_d = intensities_smoothed[d]

        # Smoothness loss
        loss = torch.sum((intensities_d - intensities_smoothed_d)**2)

        # Penalise too much deviation from the mean
        # loss += 0.1 * torch.sum((torch.log(1 + intensities_d) - torch.log(1 + intensities_d.mean().detach()))**2)

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
    losses = [torch.tensor(0., device=points[0].device)] * 3
    for d in range(3, D):
        points_d = points[d]
        points_smoothed_d = points_smoothed[d]
        loss = torch.sum((points_d - points_smoothed_d)**2)
        losses.append(loss)

    return losses


@torch.jit.script
def calculate_curvature_losses(
        points: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    The points should change smoothly along the body.
    """
    D = len(points)
    device = points[0].device
    losses = [torch.tensor(0., device=device)] * 3
    for d in range(3, D):
        loss = torch.tensor(0., device=device)
        for points_d in points[d]:
            sl = torch.norm(points_d[1:] - points_d[:-1], dim=-1)
            sp = torch.cat([torch.zeros(1, device=device), sl.cumsum(dim=0)])
            K = torch.zeros_like(points_d)
            for i in range(3):
                x = points_d[:, i]
                Tx = torch.gradient(x, spacing=(sp,))[0]
                Kx = torch.gradient(Tx, spacing=(sp,))[0]
                K[:, i] = Kx
            k = torch.norm(K, dim=-1)

            # Only penalise curvatures greater than 2-revolutions
            loss = loss + k[k > (2 * 2 * torch.pi) / sl.sum()].sum()
        losses.append(loss)

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

    # If there are no previous points available then just return zeros.
    if points_prev is None:
        return [torch.tensor(0., device=points[0].device) for _ in range(D)]

    losses = []
    for d in range(D):
        points_d = points[d]
        points_prev_d = points_prev[d]
        loss = torch.sum((points_d - points_prev_d)**2)
        losses.append(loss)

    return losses
