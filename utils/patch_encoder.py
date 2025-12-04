import torch
from typing import Tuple
def bchw_to_pixel_values(
    images: torch.Tensor,
    patch_size: int=16,
    temporal_patch_size: int=2,
    merge_size: int=2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-image preprocess and concat flatten patches.

    Args:
        images: torch.Tensor (B, C, H, W)
        patch_size: spatial patch size
        temporal_patch_size: number of temporal frames to group (>=1)
        merge_size: merge factor

    Returns:
        pixel_values: torch.Tensor (N_total_patches, vector_len)
        image_grid_thw: torch.LongTensor (B, 3) with rows [grid_t, grid_w, grid_h]
                         (matches the sample format: [1, 30, 40])
    Notes:
        - Assumes H % patch_size == 0 and W % patch_size == 0.
        - Assumes grid_h % merge_size == 0 and grid_w % merge_size == 0.
        - Each image is processed independently; temporal padding is done per-image
          by repeating the last frame to reach a multiple of temporal_patch_size.
        - All operations are differentiable (torch ops).
    """
    if images.ndim != 4:
        raise ValueError("images must have shape (B, C, H, W)")

    B, C, H, W = images.shape
    if patch_size <= 0 or temporal_patch_size <= 0 or merge_size <= 0:
        raise ValueError("patch_size, temporal_patch_size and merge_size must be > 0")

    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError("H and W must be divisible by patch_size")

    grid_h = H // patch_size
    grid_w = W // patch_size

    if grid_h % merge_size != 0 or grid_w % merge_size != 0:
        raise ValueError("grid_h and grid_w must be divisible by merge_size")

    per_image_patches = []   # list of tensors (num_patches_i, vec_len)
    grid_list = []           # list of [grid_t, grid_w, grid_h] per image (match your sample)

    # For each image in batch, treat it as a sequence of 1 frame and pad temporally to temporal_patch_size.
    for i in range(B):
        img = images[i]  # (C, H, W)

        # Make a "temporal" dimension by stacking/repeating this single frame so we have B_i frames
        # Start with single frame count = 1
        frames = img.unsqueeze(0)  # (1, C, H, W)
        if frames.shape[0] % temporal_patch_size != 0:
            pad_n = temporal_patch_size - (frames.shape[0] % temporal_patch_size)
            last = frames[-1:].expand(pad_n, -1, -1, -1).contiguous()
            frames = torch.cat([frames, last], dim=0)  # (temporal_patch_size, C, H, W)

        B_i = frames.shape[0]  # will be multiple of temporal_patch_size
        grid_t = B_i // temporal_patch_size

        # Now we have frames shaped (B_i, C, H, W)
        # reshape to expose patch dims:
        # target reshape similar to numpy code:
        # (grid_t, temporal_patch_size, C, grid_h//merge_size, merge_size, patch_size, grid_w//merge_size, merge_size, patch_size)
        # first view (grid_t, temporal_patch_size, C, grid_h, patch_size, grid_w, patch_size)
        # then split grid dims by merge_size and permute like original algorithm

        # Step1: reshape H and W into (grid_h, patch_size) and (grid_w, patch_size)
        frames_view = frames.view(
            grid_t,
            temporal_patch_size,
            C,
            grid_h,
            patch_size,
            grid_w,
            patch_size,
        )  # shape: (grid_t, t_p, C, g_h, p, g_w, p)

        # Step2: split grid_h/grid_w by merge_size: (g_h//merge_size, merge_size)
        frames_view = frames_view.view(
            grid_t,
            temporal_patch_size,
            C,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )  # (grid_t, t_p, C, gh_m, m1, p, gw_m, m2, p)

        # Permute to match numpy ordering: transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        perm = frames_view.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).contiguous()
        # shape now: (grid_t, gh_m, gw_m, m1, m2, C, t_p, p, p)

        # Combine back gh_m * m1 -> grid_h, gw_m * m2 -> grid_w
        # new view: (grid_t, grid_h, grid_w, C, t_p, p, p)
        perm_view = perm.view(
            grid_t,
            grid_h,
            grid_w,
            C,
            temporal_patch_size,
            patch_size,
            patch_size,
        ).contiguous()

        # Finally flatten patch-vector dims: (grid_t * grid_h * grid_w, C * temporal_patch_size * patch_size * patch_size)
        num_patches_i = grid_t * grid_h * grid_w
        vec_len = C * temporal_patch_size * patch_size * patch_size
        patches_i = perm_view.view(num_patches_i, vec_len)

        per_image_patches.append(patches_i)
        # IMPORTANT: user sample shows image_grid_thw as [[1,30,40]], i.e. order (grid_t, grid_w, grid_h)
        grid_list.append([grid_t, grid_w, grid_h])

    # Concatenate all images' patches along rows (first dimension)
    pixel_values = torch.cat(per_image_patches, dim=0)  # (sum num_patches_i, vec_len)
    image_grid_thw = torch.tensor(grid_list, dtype=torch.long, device=pixel_values.device)  # (B, 3)

    return pixel_values, image_grid_thw