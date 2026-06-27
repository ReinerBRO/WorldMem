import json
import os
import random
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import wandb
from torchvision.transforms import InterpolationMode
from PIL import Image
from packaging import version as pver
from einops import rearrange
from tqdm import tqdm
from omegaconf import DictConfig
from lightning.pytorch.utilities.types import STEP_OUTPUT
from algorithms.common.metrics import (
    LearnedPerceptualImagePatchSimilarity,
)
from utils.logging_utils import log_video, get_validation_metrics_for_videos
from .df_base import DiffusionForcingBase
from .models.vae import VAE_models
from .models.diffusion import Diffusion
from .models.pose_prediction import PosePredictionNet
from ptm.losses import FutureTestLoss, FutureTestLossConfig
from ptm.memory import (
    BottleneckLoss,
    FutureSupervisedVisualMemorySelector,
    FutureTestDecoder,
    PTMWorldMemAdapter,
    PredictiveTestMemory,
)
import glob

# Utility Functions
def euler_to_rotation_matrix(pitch, yaw):
    """
    Convert pitch and yaw angles (in radians) to a 3x3 rotation matrix.
    Supports batch input.

    Args:
        pitch (torch.Tensor): Pitch angles in radians.
        yaw (torch.Tensor): Yaw angles in radians.

    Returns:
        torch.Tensor: Rotation matrix of shape (batch_size, 3, 3).
    """
    cos_pitch, sin_pitch = torch.cos(pitch), torch.sin(pitch)
    cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)

    R_pitch = torch.stack([
        torch.ones_like(pitch), torch.zeros_like(pitch), torch.zeros_like(pitch),
        torch.zeros_like(pitch), cos_pitch, -sin_pitch,
        torch.zeros_like(pitch), sin_pitch, cos_pitch
    ], dim=-1).reshape(-1, 3, 3)

    R_yaw = torch.stack([
        cos_yaw, torch.zeros_like(yaw), sin_yaw,
        torch.zeros_like(yaw), torch.ones_like(yaw), torch.zeros_like(yaw),
        -sin_yaw, torch.zeros_like(yaw), cos_yaw
    ], dim=-1).reshape(-1, 3, 3)

    return torch.matmul(R_yaw, R_pitch)


def euler_to_camera_to_world_matrix(pose):
    """
    Convert (x, y, z, pitch, yaw) to a 4x4 camera-to-world transformation matrix using torch.
    Supports both (5,) and (f, b, 5) shaped inputs.

    Args:
        pose (torch.Tensor): Pose tensor of shape (5,) or (f, b, 5).

    Returns:
        torch.Tensor: Camera-to-world transformation matrix of shape (4, 4).
    """

    origin_dim = pose.ndim
    if origin_dim == 1:
        pose = pose.unsqueeze(0).unsqueeze(0)  # Convert (5,) -> (1, 1, 5)
    elif origin_dim == 2:
        pose = pose.unsqueeze(0)

    x, y, z, pitch, yaw = pose[..., 0], pose[..., 1], pose[..., 2], pose[..., 3], pose[..., 4]
    pitch, yaw = torch.deg2rad(pitch), torch.deg2rad(yaw)

    # Compute rotation matrix (batch mode)
    R = euler_to_rotation_matrix(pitch, yaw)  # Shape (f*b, 3, 3)

    # Create the 4x4 transformation matrix
    eye = torch.eye(4, dtype=torch.float32, device=pose.device)
    camera_to_world = eye.repeat(R.shape[0], 1, 1)  # Shape (f*b, 4, 4)

    # Assign rotation
    camera_to_world[:, :3, :3] = R

    # Assign translation
    camera_to_world[:, :3, 3] = torch.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], dim=-1)

    # Reshape back to (f, b, 4, 4) if needed
    if origin_dim == 3:
        return camera_to_world.view(pose.shape[0], pose.shape[1], 4, 4)
    elif origin_dim == 2:
        return camera_to_world.view(pose.shape[0], 4, 4)
    else:
        return camera_to_world.squeeze(0).squeeze(0)  # Convert (1,1,4,4) -> (4,4)

def is_inside_fov_3d_hv(points, center, center_pitch, center_yaw, fov_half_h, fov_half_v):
    """
    Check whether points are within a given 3D field of view (FOV) 
    with separately defined horizontal and vertical ranges.

    The center view direction is specified by pitch and yaw (in degrees).

    :param points: (N, B, 3) Sample point coordinates
    :param center: (3,) Center coordinates of the FOV
    :param center_pitch: Pitch angle of the center view (in degrees)
    :param center_yaw: Yaw angle of the center view (in degrees)
    :param fov_half_h: Horizontal half-FOV angle (in degrees)
    :param fov_half_v: Vertical half-FOV angle (in degrees)
    :return: Boolean tensor (N, B), indicating whether each point is inside the FOV
    """
    # Compute vectors relative to the center
    vectors = points - center  # shape (N, B, 3)
    x = vectors[..., 0]
    y = vectors[..., 1]
    z = vectors[..., 2]
    
    # Compute horizontal angle (yaw): measured with respect to the z-axis as the forward direction,
    # and the x-axis as left-right, resulting in a range of -180 to 180 degrees.
    azimuth = torch.atan2(x, z) * (180 / math.pi)
    
    # Compute vertical angle (pitch): measured with respect to the horizontal plane,
    # resulting in a range of -90 to 90 degrees.
    elevation = torch.atan2(y, torch.sqrt(x**2 + z**2)) * (180 / math.pi)
    
    # Compute the angular difference from the center view (handling circular angle wrap-around)
    diff_azimuth = (azimuth - center_yaw).abs() % 360
    diff_elevation = (elevation - center_pitch).abs() % 360
    
    # Adjust values greater than 180 degrees to the shorter angular difference
    diff_azimuth = torch.where(diff_azimuth > 180, 360 - diff_azimuth, diff_azimuth)
    diff_elevation = torch.where(diff_elevation > 180, 360 - diff_elevation, diff_elevation)
    
    # Check if both horizontal and vertical angles are within their respective FOV limits
    return (diff_azimuth < fov_half_h) & (diff_elevation < fov_half_v)
    
def generate_points_in_sphere(n_points, radius):
    # Sample three independent uniform distributions
    samples_r = torch.rand(n_points)       # For radius distribution
    samples_phi = torch.rand(n_points)     # For azimuthal angle phi
    samples_u = torch.rand(n_points)       # For polar angle theta

    # Apply cube root to ensure uniform volumetric distribution
    r = radius * torch.pow(samples_r, 1/3)
    # Azimuthal angle phi uniformly distributed in [0, 2π]
    phi = 2 * math.pi * samples_phi
    # Convert u to theta to ensure cos(theta) is uniformly distributed
    theta = torch.acos(1 - 2 * samples_u)

    # Convert spherical coordinates to Cartesian coordinates
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)

    points = torch.stack((x, y, z), dim=1)
    return points

def tensor_max_with_number(tensor, number):
    number_tensor = torch.tensor(number, dtype=tensor.dtype, device=tensor.device)
    result = torch.max(tensor, number_tensor)
    return result

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')
    
def camera_to_world_to_world_to_camera(camera_to_world: torch.Tensor) -> torch.Tensor:
    """
    Convert Camera-to-World matrices to World-to-Camera matrices for a tensor with shape (f, b, 4, 4).

    Args:
        camera_to_world (torch.Tensor): A tensor of shape (f, b, 4, 4), where:
            f = number of frames,
            b = batch size.

    Returns:
        torch.Tensor: A tensor of shape (f, b, 4, 4) representing the World-to-Camera matrices.
    """
    # Ensure input is a 4D tensor
    assert camera_to_world.ndim == 4 and camera_to_world.shape[2:] == (4, 4), \
        "Input must be of shape (f, b, 4, 4)"
    
    # Extract the rotation (R) and translation (T) parts
    R = camera_to_world[:, :, :3, :3]  # Shape: (f, b, 3, 3)
    T = camera_to_world[:, :, :3, 3]   # Shape: (f, b, 3)
    
    # Initialize an identity matrix for the output
    world_to_camera = torch.eye(4, device=camera_to_world.device).unsqueeze(0).unsqueeze(0)
    world_to_camera = world_to_camera.repeat(camera_to_world.size(0), camera_to_world.size(1), 1, 1)  # Shape: (f, b, 4, 4)
    
    # Compute the rotation (transpose of R)
    world_to_camera[:, :, :3, :3] = R.transpose(2, 3)
    
    # Compute the translation (-R^T * T)
    world_to_camera[:, :, :3, 3] = -torch.matmul(R.transpose(2, 3), T.unsqueeze(-1)).squeeze(-1)
    
    return world_to_camera.to(camera_to_world.dtype)

def convert_to_plucker(poses, curr_frame, focal_length, image_width, image_height):

    intrinsic = np.asarray([focal_length * image_width,
                                focal_length * image_height,
                                0.5 * image_width,
                                0.5 * image_height], dtype=np.float32)

    c2ws = get_relative_pose(poses, zero_first_frame_scale=curr_frame)
    c2ws = rearrange(c2ws, "t b m n -> b t m n")

    K = torch.as_tensor(intrinsic, device=poses.device, dtype=poses.dtype).repeat(c2ws.shape[0],c2ws.shape[1],1)  # [B, F, 4]
    plucker_embedding = ray_condition(K, c2ws, image_height, image_width, device=c2ws.device)
    plucker_embedding = rearrange(plucker_embedding, "b t h w d -> t b h w d").contiguous()

    return plucker_embedding


def get_relative_pose(abs_c2ws, zero_first_frame_scale):
    abs_w2cs = camera_to_world_to_world_to_camera(abs_c2ws)
    target_cam_c2w = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]).to(abs_c2ws.device).to(abs_c2ws.dtype)
    abs2rel = target_cam_c2w @ abs_w2cs[zero_first_frame_scale]
    ret_poses = [abs2rel @ abs_c2w for abs_c2w in abs_c2ws]
    ret_poses = torch.stack(ret_poses)
    return ret_poses

def ray_condition(K, c2w, H, W, device):
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B = K.shape[0]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]

    fx, fy, cx, cy = K.chunk(4, dim=-1)  # B,V, 1

    zs = torch.ones_like(i, device=device, dtype=c2w.dtype)  # [B, HxW]
    xs = -(i - cx) / fx * zs
    ys = -(j - cy) / fy * zs 

    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)  # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)  # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, 3, HW
    rays_o = c2w[..., :3, 3]  # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, 3, HW
    # c2w @ dirctions
    rays_dxo = torch.linalg.cross(rays_o, rays_d)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # B, V, H, W, 6

    return plucker

def random_transform(tensor):
    """
    Apply the same random translation, rotation, and scaling to all frames in the batch.

    Args:
        tensor (torch.Tensor): Input tensor of shape (F, B, 3, H, W).

    Returns:
        torch.Tensor: Transformed tensor of shape (F, B, 3, H, W).
    """
    if tensor.ndim != 5:
        raise ValueError("Input tensor must have shape (F, B, 3, H, W)")

    F, B, C, H, W = tensor.shape

    # Generate random transformation parameters
    max_translate = 0.2  # Translate up to 20% of width/height
    max_rotate = 30      # Rotate up to 30 degrees
    max_scale = 0.2      # Scale change by up to +/- 20%

    translate_x = random.uniform(-max_translate, max_translate) * W
    translate_y = random.uniform(-max_translate, max_translate) * H
    rotate_angle = random.uniform(-max_rotate, max_rotate)
    scale_factor = 1 + random.uniform(-max_scale, max_scale)

    # Apply the same transformation to all frames and batches

    tensor = tensor.reshape(F*B, C, H, W)
    transformed_tensor = TF.affine(
        tensor,
        angle=rotate_angle,
        translate=(translate_x, translate_y),
        scale=scale_factor,
        shear=(0, 0),
        interpolation=InterpolationMode.BILINEAR,
        fill=0
    )

    transformed_tensor = transformed_tensor.reshape(F, B, C, H, W)
    return transformed_tensor

def save_tensor_as_png(tensor, file_path):
    """
    Save a 3*H*W tensor as a PNG image.

    Args:
        tensor (torch.Tensor): Input tensor of shape (3, H, W).
        file_path (str): Path to save the PNG file.
    """
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError("Input tensor must have shape (3, H, W)")

    # Convert tensor to PIL Image
    image = TF.to_pil_image(tensor)

    # Save image
    image.save(file_path)

class WorldMemMinecraft(DiffusionForcingBase):
    """
    Video generation for MineCraft with memory.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize the WorldMemMinecraft class with the given configuration.

        Args:
            cfg (DictConfig): Configuration object.
        """
        self.n_tokens = cfg.n_frames // cfg.frame_stack # number of max tokens for the model
        self.n_frames = cfg.n_frames
        if hasattr(cfg, "n_tokens"):
            self.n_tokens = cfg.n_tokens // cfg.frame_stack
        self.memory_condition_length = cfg.memory_condition_length
        self.raw_reference_length = int(getattr(cfg, "raw_reference_length", self.memory_condition_length))
        self.pose_cond_dim = getattr(cfg, "pose_cond_dim", 5)

        self.use_plucker = getattr(cfg, "use_plucker", True)
        self.relative_embedding = getattr(cfg, "relative_embedding", True)
        self.state_embed_only_on_qk = getattr(cfg, "state_embed_only_on_qk", True)
        self.use_memory_attention = getattr(cfg, "use_memory_attention", True)
        self.use_memory_attention_runtime = bool(getattr(cfg, "use_memory_attention_runtime", self.use_memory_attention))
        self.add_timestamp_embedding = getattr(cfg, "add_timestamp_embedding", True)
        self.ref_mode = getattr(cfg, "ref_mode", 'sequential')
        self.log_curve = getattr(cfg, "log_curve", False)
        self.focal_length =  getattr(cfg, "focal_length", 0.35)
        self.log_video = cfg.log_video
        self.max_log_videos = getattr(cfg, "max_log_videos", None)
        self.video_log_stage = getattr(cfg, "video_log_stage", None)
        self.save_local = getattr(cfg, "save_local", True)
        self.local_save_dir = getattr(cfg, "local_save_dir", None)
        self.lpips_batch_size = getattr(cfg, "lpips_batch_size", 16)
        self.next_frame_length = getattr(cfg, "next_frame_length", 1)
        self.require_pose_prediction = getattr(cfg, "require_pose_prediction", False)
        self.use_ptm_memory = getattr(cfg, "use_ptm_memory", False)
        self.ptm_memory_dim = getattr(cfg, "ptm_memory_dim", 1024)
        self.ptm_num_memory_tokens = getattr(cfg, "num_memory_tokens", self.memory_condition_length)
        self.ptm_num_layers = getattr(cfg, "ptm_num_layers", 4)
        self.ptm_token_dropout = getattr(cfg, "ptm_token_dropout", 0.0)
        self.ptm_loss_weight = getattr(cfg, "ptm_loss_weight", 0.25)
        self.ptm_bottleneck_weight = getattr(cfg, "ptm_bottleneck_weight", 0.001)
        self.ptm_ablation = self._canonical_ablation_mode(getattr(cfg, "ptm_ablation", "normal"))
        self.validation_ablation_modes = self._parse_ablation_modes(
            getattr(cfg, "validation_ablation_modes", self.ptm_ablation)
        )
        self.validation_video_mode = self._canonical_ablation_mode(
            getattr(cfg, "validation_video_mode", "normal")
        )
        self.ptm_max_history = int(getattr(cfg, "ptm_max_history", self.n_frames))
        self.use_ptm_cross_attention = bool(getattr(cfg, "use_ptm_cross_attention", True)) and self.use_ptm_memory
        self.use_ptm_reference_adapter = bool(getattr(cfg, "use_ptm_reference_adapter", False)) and self.use_ptm_memory
        self.ptm_eval_only = bool(getattr(cfg, "ptm_eval_only", False)) and self.use_ptm_memory
        validation_log_step = getattr(cfg, "validation_log_step", None)
        self.validation_log_step = None if validation_log_step is None else int(validation_log_step)
        self.ptm_eval_outputs = {}
        self.ptm_context_memory_only = bool(getattr(cfg, "ptm_context_memory_only", False))
        self.ptm_train_context_memory_only = bool(getattr(cfg, "ptm_train_context_memory_only", False))
        self.ptm_train_context_token_source = str(
            getattr(cfg, "ptm_train_context_token_source", "reference_tail")
        ).strip().lower()
        self.ptm_context_memory_strategy = str(getattr(cfg, "ptm_context_memory_strategy", "strided")).strip().lower()
        if self.ptm_train_context_token_source not in {"reference_tail", "context"}:
            raise ValueError("ptm_train_context_token_source must be 'reference_tail' or 'context'")
        if self.ptm_context_memory_only or self.ptm_train_context_memory_only:
            if self.raw_reference_length != 0:
                raise ValueError("PTM context-memory-only modes require raw_reference_length=0")
            if self.use_memory_attention_runtime:
                raise ValueError("PTM context-memory-only modes require use_memory_attention_runtime=false")
            if self.use_ptm_reference_adapter:
                raise ValueError("PTM context-memory-only modes require use_ptm_reference_adapter=false")
            if self.ptm_context_memory_strategy not in {"strided", "recent"}:
                raise ValueError("ptm_context_memory_strategy must be 'strided' or 'recent'")
        if self.ptm_train_context_memory_only:
            if not self.use_ptm_memory:
                raise ValueError("ptm_train_context_memory_only requires use_ptm_memory=true")
        self.generation_target_window_radius = int(getattr(cfg, "generation_target_window_radius", 5))
        self.generation_late_horizon_start = int(getattr(cfg, "generation_late_horizon_start", 50))
        self.generation_target_loss_weight = float(getattr(cfg, "generation_target_loss_weight", 0.0))
        self.generation_late_loss_weight = float(getattr(cfg, "generation_late_loss_weight", 0.0))
        self.generation_wandb_detailed_metrics = bool(getattr(cfg, "generation_wandb_detailed_metrics", False))
        self._generation_compare_history = {}
        # PTM-as-generation-consumer flags.
        # When True, PTM tokens fed to DiT are detached so diffusion loss only
        # trains the DiT consumer (cross-attn), not the PTM encoder. The PTM
        # encoder is trained solely by the future-test loss.
        self.ptm_detach_for_generation = bool(getattr(cfg, "ptm_detach_for_generation", False))
        self.ptm_contrast_weight = float(getattr(cfg, "ptm_contrast_weight", 0.0))
        self.ptm_contrast_margin = float(getattr(cfg, "ptm_contrast_margin", 0.0))
        # When True, freeze the DiT backbone and only train ptm_* params inside
        # DiT (ptm_memory_proj / ptm_norm / ptm_attn / ptm_gate) plus the PTM
        # encoder and test decoder. Protects WorldMem generation capability.
        self.ptm_train_consumer_only = bool(getattr(cfg, "ptm_train_consumer_only", False))
        self.ptm_visual_memory_selection = bool(getattr(cfg, "ptm_visual_memory_selection", False)) and self.use_ptm_memory
        self.ptm_visual_top_k = int(getattr(cfg, "ptm_visual_top_k", 8))
        self.ptm_visual_num_candidates = int(
            getattr(cfg, "ptm_visual_num_candidates", getattr(cfg, "ptm_max_history_candidates", self.n_frames))
        )
        self.ptm_visual_pool = str(getattr(cfg, "ptm_visual_pool", "grid2x2")).strip().lower()
        self.ptm_visual_candidate_source = str(getattr(cfg, "ptm_visual_candidate_source", "context_strided")).strip().lower()
        self.ptm_visual_include_summary_tokens = bool(getattr(cfg, "ptm_visual_include_summary_tokens", True))
        self.ptm_visual_remap_match_labels = bool(getattr(cfg, "ptm_visual_remap_match_labels", True))
        if self.ptm_visual_memory_selection:
            if self.ptm_visual_top_k <= 0:
                raise ValueError("ptm_visual_top_k must be positive")
            if self.ptm_visual_num_candidates <= 0:
                raise ValueError("ptm_visual_num_candidates must be positive")
            if self.ptm_visual_pool not in {"global", "grid2x2"}:
                raise ValueError("ptm_visual_pool must be 'global' or 'grid2x2'")
            if self.ptm_visual_candidate_source not in {"batch", "context_strided", "context_recent"}:
                raise ValueError(
                    "ptm_visual_candidate_source must be 'batch', 'context_strided', or 'context_recent'"
                )

        super().__init__(cfg)

    def _canonical_ablation_mode(self, mode: str) -> str:
        normalized = str(mode).strip().lower().replace("-", "_")
        if normalized == "shuffle":
            normalized = "hard_shuffle"
        if normalized in {"shuffle_token", "token_shuffle"}:
            normalized = "shuffle_token"
        if normalized in {"zero_token", "token_zero"}:
            normalized = "zero_token"
        if normalized not in {"normal", "zero", "hard_shuffle", "zero_token", "shuffle_token"}:
            raise ValueError(f"unknown PTM ablation mode: {mode}")
        return normalized

    def _parse_ablation_modes(self, value) -> list[str]:
        if isinstance(value, str):
            parts = [part for part in value.replace(";", ",").split(",") if part.strip()]
        else:
            parts = list(value)
        modes = []
        for part in parts:
            mode = self._canonical_ablation_mode(part)
            if mode not in modes:
                modes.append(mode)
        if not modes:
            raise ValueError("validation_ablation_modes must contain at least one mode")
        return modes

    def _video_log_namespace(self, namespace: str) -> str:
        base_namespace = namespace + "_vis"
        if not self.video_log_stage:
            return base_namespace
        stage = str(self.video_log_stage).strip().lower().replace("-", "_")
        return f"{base_namespace}_{stage}"

    def _validation_log_step(self, namespace: str) -> int | None:
        if namespace == "test":
            return None
        return self.validation_log_step if self.validation_log_step is not None else int(self.global_step)
            
    def _build_model(self):

        self.diffusion_model = Diffusion(
            reference_length=self.raw_reference_length,
            x_shape=self.x_stacked_shape,
            action_cond_dim=self.action_cond_dim,
            pose_cond_dim=self.pose_cond_dim,
            is_causal=self.causal,
            cfg=self.cfg.diffusion,
            is_dit=True,
            use_plucker=self.use_plucker,
            relative_embedding=self.relative_embedding,
            state_embed_only_on_qk=self.state_embed_only_on_qk,
            use_memory_attention=self.use_memory_attention,
            add_timestamp_embedding=self.add_timestamp_embedding,
            ref_mode=self.ref_mode,
            use_ptm_cross_attention=self.use_ptm_cross_attention,
            ptm_memory_dim=self.ptm_memory_dim,
        )
        if not self.use_memory_attention_runtime:
            self.diffusion_model.use_memory_attention = False
            if hasattr(self.diffusion_model, "model"):
                self.diffusion_model.model.use_memory_attention = False
                if hasattr(self.diffusion_model.model, "blocks"):
                    for block in self.diffusion_model.model.blocks:
                        if hasattr(block, "use_memory_attention"):
                            block.use_memory_attention = False

        self.validation_lpips_model = LearnedPerceptualImagePatchSimilarity()
        vae = VAE_models["vit-l-20-shallow-encoder"]()
        self.vae = vae.eval()

        if self.require_pose_prediction:
            self.pose_prediction_model = PosePredictionNet()

        if self.use_ptm_memory:
            if self.memory_condition_length <= 0:
                raise ValueError("use_ptm_memory requires memory_condition_length > 0")
            self.ptm_memory = PredictiveTestMemory(
                frame_dim=self.vae.latent_dim,
                action_dim=self.action_cond_dim,
                pose_dim=self.pose_cond_dim,
                memory_dim=self.ptm_memory_dim,
                num_memory_tokens=self.ptm_num_memory_tokens,
                num_layers=self.ptm_num_layers,
                max_history=self.ptm_max_history,
                token_dropout=self.ptm_token_dropout,
            )
            if self.use_ptm_reference_adapter:
                self.ptm_worldmem_adapter = PTMWorldMemAdapter(
                    memory_dim=self.ptm_memory_dim,
                    latent_channels=self.vae.latent_dim,
                    latent_height=self.vae.seq_h,
                    latent_width=self.vae.seq_w,
                    action_dim=self.action_cond_dim,
                    pose_dim=self.pose_cond_dim,
                )
            self.ptm_test_decoder = FutureTestDecoder(
                memory_dim=self.ptm_memory_dim,
                action_dim=self.action_cond_dim,
                future_embedding_dim=self.vae.latent_dim,
                max_history_candidates=getattr(self.cfg, "ptm_max_history_candidates", self.n_frames),
            )
            if self.ptm_visual_memory_selection:
                self.ptm_visual_selector = FutureSupervisedVisualMemorySelector(
                    frame_dim=self.vae.latent_dim,
                    memory_dim=self.ptm_memory_dim,
                    top_k=self.ptm_visual_top_k,
                    pool=self.ptm_visual_pool,
                    dropout=getattr(self.cfg, "ptm_visual_dropout", 0.0),
                )
            self.ptm_test_loss = FutureTestLoss(
                FutureTestLossConfig(
                    w_embed=getattr(self.cfg, "ptm_w_embed", 1.0),
                    w_loop=getattr(self.cfg, "ptm_w_loop", 0.5),
                    w_match=getattr(self.cfg, "ptm_w_match", 0.5),
                    w_landmark=getattr(self.cfg, "ptm_w_landmark", 0.5),
                    w_object=getattr(self.cfg, "ptm_w_object", 0.5),
                )
            )
            self.ptm_bottleneck_loss = BottleneckLoss(l2_weight=self.ptm_bottleneck_weight)

    def configure_optimizers(self):
        if self.ptm_train_consumer_only:
            # Freeze the WorldMem/DiT backbone; only train ptm_* params inside
            # DiT (ptm_memory_proj / ptm_norm / ptm_attn / ptm_gate) so the
            # generation capability is preserved while the PTM consumer learns.
            for p in self.diffusion_model.parameters():
                p.requires_grad_(False)
            consumer_params = []
            for name, p in self.diffusion_model.named_parameters():
                if "ptm_" in name:
                    p.requires_grad_(True)
                    consumer_params.append(p)
            params = list(consumer_params)
        else:
            params = list(self.diffusion_model.parameters())
        if self.use_ptm_memory:
            params += list(self.ptm_memory.parameters())
            if self.use_ptm_reference_adapter:
                params += list(self.ptm_worldmem_adapter.parameters())
            if self.ptm_visual_memory_selection:
                params += list(self.ptm_visual_selector.parameters())
            params += list(self.ptm_test_decoder.parameters())
        optimizer_dynamics = torch.optim.AdamW(
            params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay, betas=self.cfg.optimizer_beta
        )
        return optimizer_dynamics

    def _extract_ptm_supervision(self, batch):
        if not isinstance(batch, dict):
            return None
        required = {"future_actions", "memory_labels", "target_frames"}
        if not required.issubset(batch):
            return None
        return {
            "future_actions": batch["future_actions"],
            "memory_labels": batch["memory_labels"],
            "target_frames": batch["target_frames"],
        }

    def _visual_candidate_indices(
        self,
        xs: torch.Tensor,
        batch: dict | None,
        recent_end: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return candidate local indices as [N,B] plus valid mask [B,N]."""
        batch_size = xs.shape[1]
        device = xs.device
        max_candidates = max(1, int(self.ptm_visual_num_candidates))
        recent_end = max(0, min(int(recent_end), int(xs.shape[0])))

        if (
            self.ptm_visual_candidate_source == "batch"
            and isinstance(batch, dict)
            and "candidate_history_indices" in batch
        ):
            candidate = batch["candidate_history_indices"]
            if not torch.is_tensor(candidate):
                candidate = torch.as_tensor(candidate)
            candidate = candidate.to(device=device, dtype=torch.long)
            if candidate.ndim == 1:
                candidate = candidate[None].expand(batch_size, -1)
            elif candidate.ndim == 2 and candidate.shape[0] != batch_size and candidate.shape[1] == batch_size:
                candidate = candidate.transpose(0, 1)
            if candidate.ndim != 2 or candidate.shape[0] != batch_size:
                raise ValueError(
                    "candidate_history_indices must be [B,N] or [N,B], "
                    f"got {tuple(candidate.shape)} for batch_size={batch_size}"
                )
            candidate = candidate[:, :max_candidates].clamp(min=0, max=max(0, xs.shape[0] - 1))
            counts = batch.get("candidate_history_count") if isinstance(batch, dict) else None
            if counts is None:
                counts = torch.full((batch_size,), candidate.shape[1], device=device, dtype=torch.long)
            elif torch.is_tensor(counts):
                counts = counts.to(device=device, dtype=torch.long).flatten()
            else:
                counts = torch.as_tensor(counts, device=device, dtype=torch.long).flatten()
            if counts.numel() == 1 and batch_size > 1:
                counts = counts.expand(batch_size)
            counts = counts[:batch_size].clamp(min=0, max=candidate.shape[1])
            positions = torch.arange(candidate.shape[1], device=device)[None]
            mask = positions < counts[:, None]
            return candidate.transpose(0, 1).contiguous(), mask

        if recent_end <= 0:
            candidate = torch.zeros((1, batch_size), device=device, dtype=torch.long)
            mask = torch.zeros((batch_size, 1), device=device, dtype=torch.bool)
            return candidate, mask

        count = min(max_candidates, recent_end)
        if self.ptm_visual_candidate_source == "context_recent":
            base = torch.arange(recent_end - count, recent_end, device=device, dtype=torch.long)
        else:
            base = torch.linspace(0, recent_end - 1, steps=count, device=device).round().to(torch.long)
        candidate = base[:, None].expand(-1, batch_size).contiguous()
        mask = torch.ones((batch_size, candidate.shape[0]), device=device, dtype=torch.bool)
        return candidate, mask

    def _visual_candidate_latents(self, xs: torch.Tensor, candidate_indices: torch.Tensor) -> torch.Tensor:
        candidate_latents = self._gather_time_batch(xs, candidate_indices)
        return candidate_latents.permute(1, 0, 2, 3, 4).contiguous().to(self.device)

    def _visual_labels_for_candidates(
        self,
        labels: dict[str, torch.Tensor],
        candidate_indices: torch.Tensor,
        candidate_mask: torch.Tensor,
        batch: dict | None,
    ) -> dict[str, torch.Tensor]:
        if not self.ptm_visual_remap_match_labels:
            return labels
        if not isinstance(batch, dict) or "timestamp" not in batch or "matched_history_t" not in batch:
            return labels
        timestamp = batch["timestamp"]
        if not torch.is_tensor(timestamp):
            timestamp = torch.as_tensor(timestamp)
        timestamp = timestamp.to(device=candidate_indices.device, dtype=torch.long)
        if timestamp.ndim != 2:
            return labels
        if timestamp.shape[0] != candidate_mask.shape[0] and timestamp.shape[1] == candidate_mask.shape[0]:
            timestamp = timestamp.transpose(0, 1)
        if timestamp.shape[0] != candidate_mask.shape[0]:
            return labels

        matched_t = batch["matched_history_t"]
        if not torch.is_tensor(matched_t):
            matched_t = torch.as_tensor(matched_t)
        matched_t = matched_t.to(device=candidate_indices.device, dtype=torch.long).flatten()
        if matched_t.numel() == 1 and candidate_mask.shape[0] > 1:
            matched_t = matched_t.expand(candidate_mask.shape[0])
        if matched_t.numel() != candidate_mask.shape[0]:
            return labels

        local = candidate_indices.transpose(0, 1).clamp(min=0, max=timestamp.shape[1] - 1)
        candidate_times = timestamp.gather(dim=1, index=local)
        valid_target = matched_t[:, None] >= 0
        matches = (candidate_times == matched_t[:, None]) & candidate_mask & valid_target
        remapped = dict(labels)
        match_valid = matches.any(dim=1)
        remapped["matched_history_index"] = torch.where(
            match_valid,
            matches.to(torch.float32).argmax(dim=1).to(torch.long),
            torch.zeros_like(matched_t, dtype=torch.long),
        )
        remapped["match_valid"] = match_valid
        return remapped

    def _build_visual_condition_tokens(
        self,
        memory_tokens: torch.Tensor,
        xs: torch.Tensor,
        batch: dict | None,
        ptm_supervision: dict | None,
        recent_end: int,
    ) -> tuple[torch.Tensor, dict | None]:
        if not self.ptm_visual_memory_selection or memory_tokens is None:
            return memory_tokens, None
        if ptm_supervision is None:
            raise ValueError("ptm_visual_memory_selection requires future_actions/memory_labels in the batch")

        candidate_indices, candidate_mask = self._visual_candidate_indices(xs, batch, recent_end)
        candidate_latents = self._visual_candidate_latents(xs, candidate_indices)
        candidate_embeddings = self.ptm_visual_selector.candidate_embeddings(candidate_latents)
        labels = self._normalize_ptm_labels(ptm_supervision["memory_labels"], memory_tokens.device)
        labels = self._visual_labels_for_candidates(labels, candidate_indices, candidate_mask, batch)
        future_actions = ptm_supervision["future_actions"].to(memory_tokens.device)
        predictions = self.ptm_test_decoder(
            memory_tokens,
            future_actions,
            labels["test_type_id"],
            candidate_history_embeddings=candidate_embeddings,
        )
        visual_tokens = self.ptm_visual_selector.selected_visual_tokens(
            candidate_latents,
            predictions["match_history_logits"],
            candidate_mask=candidate_mask.to(memory_tokens.device),
            top_k=self.ptm_visual_top_k,
        )
        if self.ptm_visual_include_summary_tokens:
            condition_tokens = torch.cat([memory_tokens, visual_tokens], dim=1)
        else:
            condition_tokens = visual_tokens
        aux = {
            "candidate_embeddings": candidate_embeddings,
            "candidate_indices": candidate_indices,
            "candidate_mask": candidate_mask,
            "predictions": predictions,
            "labels": labels,
            "visual_tokens": visual_tokens,
        }
        return condition_tokens, aux

    def _batch_has_reference_tail(self, batch) -> bool:
        if not isinstance(batch, dict) or "has_reference_tail" not in batch:
            return False
        value = batch["has_reference_tail"]
        if torch.is_tensor(value):
            return bool(value.detach().flatten().any().item())
        if isinstance(value, (list, tuple)):
            return any(bool(item) for item in value)
        return bool(value)

    def _encode_ptm_memory(
        self,
        xs: torch.Tensor,
        conditions: torch.Tensor,
        pose_conditions: torch.Tensor | None = None,
        exclude_reference_tail: bool = True,
        query_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if exclude_reference_tail and self.memory_condition_length and xs.shape[0] > self.memory_condition_length:
            main_xs = xs[:-self.memory_condition_length]
            main_actions = conditions[:-self.memory_condition_length]
            reference_xs = xs[-self.memory_condition_length:]
            reference_actions = conditions[-self.memory_condition_length:]
            main_poses = pose_conditions[:-self.memory_condition_length] if pose_conditions is not None else None
            reference_poses = pose_conditions[-self.memory_condition_length:] if pose_conditions is not None else None
        else:
            main_xs = xs
            main_actions = conditions
            reference_xs = None
            reference_actions = None
            main_poses = pose_conditions
            reference_poses = None

        key_padding_mask = None
        if query_indices is not None:
            query_indices = query_indices.to(device=xs.device, dtype=torch.long).flatten()
            if query_indices.numel() == 1 and xs.shape[1] > 1:
                query_indices = query_indices.expand(xs.shape[1])
            if query_indices.numel() != xs.shape[1]:
                raise ValueError(
                    f"query_indices must have batch size {xs.shape[1]}, got {query_indices.numel()}"
                )
            query_indices = query_indices.clamp(min=0, max=main_xs.shape[0] - 1)
            prefix_len = int(query_indices.max().item()) + 1
            history_xs = main_xs[:prefix_len].clone()
            history_actions = main_actions[:prefix_len].clone()
            history_poses = main_poses[:prefix_len].clone() if main_poses is not None else None
            valid = torch.arange(prefix_len, device=xs.device)[:, None] <= query_indices[None]
            history_xs = history_xs * valid[:, :, None, None, None].to(history_xs.dtype)
            history_actions = history_actions * valid[:, :, None].to(history_actions.dtype)
            if history_poses is not None:
                history_poses = history_poses * valid[:, :, None].to(history_poses.dtype)
            key_padding_mask = ~valid.transpose(0, 1)
            if reference_xs is not None:
                history_xs = torch.cat([reference_xs, history_xs], dim=0)
                history_actions = torch.cat([reference_actions, history_actions], dim=0)
                if history_poses is not None and reference_poses is not None:
                    history_poses = torch.cat([reference_poses, history_poses], dim=0)
                reference_mask = torch.zeros(
                    xs.shape[1],
                    reference_xs.shape[0],
                    device=xs.device,
                    dtype=torch.bool,
                )
                key_padding_mask = torch.cat([reference_mask, key_padding_mask], dim=1)
        else:
            history_xs = main_xs
            history_actions = main_actions
            history_poses = main_poses
        return self.ptm_memory(
            history_xs.permute(1, 0, 2, 3, 4).contiguous(),
            history_actions.permute(1, 0, 2).contiguous(),
            pose_tokens=history_poses.permute(1, 0, 2).contiguous() if history_poses is not None else None,
            key_padding_mask=key_padding_mask,
        )

    def _apply_memory_ablation(self, memory_tokens: torch.Tensor, mode: str | None = None) -> torch.Tensor:
        mode = self._canonical_ablation_mode(mode or self.ptm_ablation)
        if mode in {"normal", "zero", "hard_shuffle", "zero_token", "shuffle_token"}:
            return memory_tokens
        raise ValueError(f"unknown PTM memory ablation mode: {mode}")

    def _batch_int(self, batch: dict | None, key: str, default: int) -> int:
        if not isinstance(batch, dict) or key not in batch:
            return int(default)
        value = batch[key]
        if torch.is_tensor(value):
            flat = value.detach().flatten()
            if flat.numel() == 0:
                return int(default)
            return int(flat[0].item())
        if isinstance(value, (list, tuple)):
            return int(value[0]) if value else int(default)
        return int(value)

    def _reference_tail_indices(self, xs: torch.Tensor, batch: dict | None) -> torch.Tensor:
        batch_size = xs.shape[1]
        memory_length = self._batch_int(batch, "memory_condition_length", self.memory_condition_length)
        if memory_length <= 0 or not self._batch_has_reference_tail(batch):
            return torch.empty((0, batch_size), dtype=torch.long, device=xs.device)
        start = xs.shape[0] - memory_length
        if start < 0:
            raise ValueError(f"invalid PTM reference tail length {memory_length} for sequence length {xs.shape[0]}")
        return torch.arange(start, xs.shape[0], device=xs.device, dtype=torch.long)[:, None].expand(-1, batch_size)

    def _context_memory_indices(self, context_length: int, batch_size: int, device: torch.device) -> torch.Tensor:
        memory_length = int(self.memory_condition_length)
        if memory_length <= 0 or context_length <= 0:
            return torch.empty((0, batch_size), dtype=torch.long, device=device)
        if self.ptm_context_memory_strategy == "recent":
            start = max(0, context_length - memory_length)
            indices = torch.arange(start, context_length, dtype=torch.long, device=device)
            if indices.numel() < memory_length:
                pad = indices[:1].expand(memory_length - indices.numel()) if indices.numel() else torch.zeros(
                    memory_length, dtype=torch.long, device=device
                )
                indices = torch.cat([pad, indices], dim=0)
        else:
            indices = torch.linspace(
                0,
                context_length - 1,
                steps=memory_length,
                device=device,
            ).round().to(dtype=torch.long)
        return indices[:, None].expand(-1, batch_size)

    def _recent_end_from_batch(self, xs: torch.Tensor, batch: dict | None) -> int:
        if not isinstance(batch, dict) or "ptm_recent_end_index" not in batch:
            raise ValueError("PTM training/eval batch must provide ptm_recent_end_index")
        recent_end = batch["ptm_recent_end_index"]
        if not torch.is_tensor(recent_end):
            recent_end = torch.as_tensor(recent_end)
        recent_end = recent_end.detach().flatten().to(dtype=torch.long)
        if recent_end.numel() == 0:
            raise ValueError("ptm_recent_end_index is empty")
        value = int(recent_end.min().item())
        main_length = xs.shape[0] - self.memory_condition_length if self._batch_has_reference_tail(batch) else xs.shape[0]
        if value < 0 or value > main_length:
            raise ValueError(f"invalid ptm_recent_end_index={value} for main length {main_length}")
        return value

    def build_ptm_memory_input(
        self,
        xs: torch.Tensor,
        actions: torch.Tensor,
        poses: torch.Tensor,
        selected_indices: torch.Tensor | np.ndarray,
        recent_start: int,
        recent_end: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        selected_indices = torch.as_tensor(selected_indices, dtype=torch.long, device=xs.device)
        if selected_indices.ndim == 1:
            selected_xs = xs[selected_indices]
            selected_actions = actions[selected_indices.to(actions.device)]
            selected_poses = poses[selected_indices.to(poses.device)]
        elif selected_indices.ndim == 2:
            selected_xs = self._gather_time_batch(xs, selected_indices)
            selected_actions = self._gather_time_batch(actions, selected_indices)
            selected_poses = self._gather_time_batch(poses, selected_indices)
        else:
            raise ValueError(f"selected_indices must be [M] or [M,B], got shape {tuple(selected_indices.shape)}")

        recent_xs = xs[recent_start:recent_end]
        recent_actions = actions[recent_start:recent_end]
        recent_poses = poses[recent_start:recent_end]

        # Pre-truncate recent so selected memory is never dropped by the
        # PredictiveTestMemory max_history truncation. Keep the most recent
        # (max_history - selected_count) frames so the encoder always sees the
        # selected reference tail plus the most recent context.
        selected_count = int(selected_indices.shape[0]) if selected_indices.dim() > 0 else 0
        if selected_count > 0:
            recent_budget = max(0, int(self.ptm_max_history) - selected_count)
            if recent_xs.shape[0] > recent_budget:
                recent_xs = recent_xs[-recent_budget:] if recent_budget > 0 else recent_xs[:0]
                recent_actions = recent_actions[-recent_budget:] if recent_budget > 0 else recent_actions[:0]
                recent_poses = recent_poses[-recent_budget:] if recent_budget > 0 else recent_poses[:0]

        return (
            torch.cat([selected_xs, recent_xs], dim=0),
            torch.cat([selected_actions, recent_actions], dim=0),
            torch.cat([selected_poses, recent_poses], dim=0),
        )

    def _hard_shuffle_order(
        self,
        batch_size: int,
        device: torch.device,
        episode_dirs: list[str] | None = None,
        episode_families: list[str] | None = None,
    ) -> torch.Tensor:
        if batch_size <= 1:
            raise ValueError("PTM hard-shuffle requires batch_size > 1")
        if episode_dirs is None or len(episode_dirs) != batch_size:
            raise ValueError("PTM hard-shuffle requires episode_dir metadata for every sample")
        order: list[int] = []
        for i in range(batch_size):
            candidates = [j for j in range(batch_size) if str(episode_dirs[j]) != str(episode_dirs[i])]
            if not candidates:
                raise ValueError("PTM hard-shuffle requires at least two different episodes in the batch")
            if episode_families is not None and len(episode_families) == batch_size and episode_families[i]:
                family_candidates = [
                    j for j in candidates if episode_families[j] and str(episode_families[j]) != str(episode_families[i])
                ]
                if family_candidates:
                    candidates = family_candidates
            order.append(candidates[i % len(candidates)])
        return torch.tensor(order, dtype=torch.long, device=device)

    def _validation_shuffle_tokens_by_episode(
        self,
        tokens: torch.Tensor,
        batch: dict | None,
    ) -> torch.Tensor:
        """Shuffle token conditioning using different episodes across the DDP global batch."""
        if tokens.shape[0] <= 1:
            raise ValueError("PTM shuffle_token validation requires batch_size > 1")
        episode_ids = batch.get("episode_id") if isinstance(batch, dict) else None
        if episode_ids is None:
            episode_dirs = batch.get("episode_dir") if isinstance(batch, dict) else None
            episode_families = batch.get("episode_family") if isinstance(batch, dict) else None
            order = self._hard_shuffle_order(tokens.shape[0], tokens.device, episode_dirs, episode_families)
            return tokens[order]
        episode_ids = torch.as_tensor(episode_ids, dtype=torch.long, device=tokens.device)
        if episode_ids.numel() != tokens.shape[0]:
            raise ValueError(
                f"episode_id batch size {episode_ids.numel()} does not match token batch size {tokens.shape[0]}"
            )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            gathered_tokens = [torch.empty_like(tokens) for _ in range(world_size)]
            torch.distributed.all_gather(gathered_tokens, tokens.contiguous())
            gathered_episode_ids = [torch.empty_like(episode_ids) for _ in range(world_size)]
            torch.distributed.all_gather(gathered_episode_ids, episode_ids.contiguous())
            all_tokens = torch.cat(gathered_tokens, dim=0)
            all_episode_ids = torch.cat(gathered_episode_ids, dim=0)
            local_batch = tokens.shape[0]
            selected = []
            for local_idx in range(local_batch):
                global_idx = rank * local_batch + local_idx
                candidates = torch.nonzero(all_episode_ids != episode_ids[local_idx], as_tuple=False).flatten()
                if candidates.numel() == 0:
                    raise ValueError("PTM shuffle_token validation requires at least two episodes in the DDP batch")
                selected.append(candidates[global_idx % candidates.numel()])
            selected_indices = torch.stack(selected).to(tokens.device)
            return all_tokens.index_select(0, selected_indices)

        candidates_by_item = []
        for item_episode in episode_ids:
            candidates = torch.nonzero(episode_ids != item_episode, as_tuple=False).flatten()
            if candidates.numel() == 0:
                raise ValueError("PTM shuffle_token validation requires at least two episodes in the batch")
            candidates_by_item.append(candidates)
        selected = [
            candidates[int(local_idx) % candidates.numel()]
            for local_idx, candidates in enumerate(candidates_by_item)
        ]
        return tokens.index_select(0, torch.stack(selected).to(tokens.device))

    def _apply_ptm_input_ablation(
        self,
        xs: torch.Tensor,
        actions: torch.Tensor,
        poses: torch.Tensor,
        selected_count: int,
        batch: dict | None = None,
        mode: str | None = None,
        ablate_tail: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Ablate memory input. By default ablates the first `selected_count` frames
        (the reference tail prefix). If `ablate_tail > 0`, ablates the last
        `ablate_tail` frames instead (the recent-history suffix the encoder
        actually sees after max_history truncation)."""
        mode = self._canonical_ablation_mode(mode or self.ptm_ablation)
        # token-level modes don't apply to input ablation; treat as normal here.
        if mode in {"zero_token", "shuffle_token"}:
            mode = "normal"
        if ablate_tail > 0:
            # Ablate the tail (recent frames the encoder actually consumes).
            n = xs.shape[0]
            if mode == "normal" or ablate_tail >= n:
                return xs, actions, poses
            xs = xs.clone()
            actions = actions.clone()
            poses = poses.clone()
            tail_start = n - ablate_tail
            if mode == "zero":
                xs[tail_start:] = 0
                actions[tail_start:] = 0
                poses[tail_start:] = 0
                return xs, actions, poses
            if mode == "hard_shuffle":
                episode_dirs = batch.get("episode_dir") if isinstance(batch, dict) else None
                episode_families = batch.get("episode_family") if isinstance(batch, dict) else None
                order = self._hard_shuffle_order(xs.shape[1], xs.device, episode_dirs, episode_families)
                xs[tail_start:] = xs[tail_start:, order]
                actions[tail_start:] = actions[tail_start:, order]
                poses[tail_start:] = poses[tail_start:, order]
                return xs, actions, poses
            raise ValueError(f"unknown PTM memory ablation mode: {mode}")
        if mode == "normal" or selected_count <= 0:
            return xs, actions, poses
        xs = xs.clone()
        actions = actions.clone()
        poses = poses.clone()
        if mode == "zero":
            xs[:selected_count] = 0
            actions[:selected_count] = 0
            poses[:selected_count] = 0
            return xs, actions, poses
        if mode == "hard_shuffle":
            episode_dirs = batch.get("episode_dir") if isinstance(batch, dict) else None
            episode_families = batch.get("episode_family") if isinstance(batch, dict) else None
            order = self._hard_shuffle_order(xs.shape[1], xs.device, episode_dirs, episode_families)
            xs[:selected_count] = xs[:selected_count, order]
            actions[:selected_count] = actions[:selected_count, order]
            poses[:selected_count] = poses[:selected_count, order]
            return xs, actions, poses
        raise ValueError(f"unknown PTM memory ablation mode: {mode}")

    def _encode_ptm_memory_input(
        self,
        xs: torch.Tensor,
        actions: torch.Tensor,
        poses: torch.Tensor,
        selected_indices: torch.Tensor | np.ndarray,
        recent_start: int,
        recent_end: int,
        batch: dict | None = None,
        ablation_mode: str | None = None,
        ablate_tail: int = 0,
    ) -> torch.Tensor | None:
        ptm_xs, ptm_actions, ptm_poses = self.build_ptm_memory_input(
            xs,
            actions,
            poses,
            selected_indices,
            recent_start,
            recent_end,
        )
        ptm_actions = ptm_actions.to(ptm_xs.device)
        ptm_poses = ptm_poses.to(ptm_xs.device)
        if ptm_xs.shape[0] == 0:
            return None
        selected_count = int(torch.as_tensor(selected_indices).shape[0])
        ptm_xs, ptm_actions, ptm_poses = self._apply_ptm_input_ablation(
            ptm_xs,
            ptm_actions,
            ptm_poses,
            selected_count,
            batch,
            mode=ablation_mode,
            ablate_tail=ablate_tail,
        )
        return self._encode_ptm_memory(
            ptm_xs.to(self.device),
            ptm_actions.to(self.device),
            ptm_poses.to(self.device),
            exclude_reference_tail=False,
        )

    def _fit_reference_length(self, tensor: torch.Tensor, reference_length: int) -> torch.Tensor:
        if tensor.shape[0] == reference_length:
            return tensor
        if tensor.shape[0] > reference_length:
            return tensor[:reference_length]
        pad_shape = (reference_length - tensor.shape[0],) + tuple(tensor.shape[1:])
        padding = torch.zeros(pad_shape, device=tensor.device, dtype=tensor.dtype)
        return torch.cat([tensor, padding], dim=0)

    def _ptm_reference_latents(
        self,
        memory_tokens: torch.Tensor,
        reference_length: int,
        dtype: torch.dtype,
        spatial_size: tuple[int, int],
    ) -> torch.Tensor:
        references = self.ptm_worldmem_adapter(self._apply_memory_ablation(memory_tokens))
        ref_latents = self._fit_reference_length(
            references["latent_reference_frames"].to(dtype),
            reference_length,
        )
        if ref_latents.shape[-2:] != spatial_size:
            ref_shape = ref_latents.shape
            ref_latents = F.interpolate(
                ref_latents.reshape(ref_shape[0] * ref_shape[1], *ref_shape[2:]),
                size=spatial_size,
                mode="bilinear",
                align_corners=False,
            ).reshape(ref_shape[0], ref_shape[1], ref_shape[2], spatial_size[0], spatial_size[1])
        return ref_latents

    def _apply_ptm_memory_references(
        self,
        xs: torch.Tensor,
        conditions: torch.Tensor,
        pose_conditions: torch.Tensor | None = None,
        memory_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Replace WorldMem reference latents with PTM pseudo references."""
        if not self.use_ptm_reference_adapter or not self.memory_condition_length:
            return xs, conditions, memory_tokens

        if memory_tokens is None:
            raise ValueError("PTM reference adapter requires memory tokens built by build_ptm_memory_input")
        ref_latents = self._ptm_reference_latents(
            memory_tokens,
            self.memory_condition_length,
            xs.dtype,
            xs.shape[-2:],
        )
        xs = xs.clone()
        xs[-self.memory_condition_length:] = ref_latents
        return xs, conditions, memory_tokens

    def _normalize_ptm_labels(self, memory_labels: dict, device: torch.device) -> dict[str, torch.Tensor]:
        labels = {}
        for key, value in memory_labels.items():
            if torch.is_tensor(value):
                labels[key] = value.to(device)
            else:
                dtype = torch.long if key in {"test_type_id", "matched_history_index"} else torch.float32
                labels[key] = torch.as_tensor(value, dtype=dtype, device=device)
        return labels

    def _ptm_predictions(
        self,
        memory_tokens: torch.Tensor,
        ptm_supervision: dict,
        candidate_history_embeddings: torch.Tensor | None = None,
        labels_override: dict[str, torch.Tensor] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        labels = labels_override or self._normalize_ptm_labels(ptm_supervision["memory_labels"], memory_tokens.device)
        future_actions = ptm_supervision["future_actions"].to(memory_tokens.device)
        predictions = self.ptm_test_decoder(
            memory_tokens,
            future_actions,
            labels["test_type_id"],
            candidate_history_embeddings=candidate_history_embeddings,
        )
        return predictions, labels

    def _compute_ptm_test_loss(
        self,
        memory_tokens: torch.Tensor,
        ptm_supervision: dict,
        visual_aux: dict | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if visual_aux is not None and "predictions" in visual_aux and "labels" in visual_aux:
            predictions = visual_aux["predictions"]
            labels = visual_aux["labels"]
        else:
            predictions, labels = self._ptm_predictions(
                memory_tokens,
                ptm_supervision,
                candidate_history_embeddings=visual_aux.get("candidate_embeddings") if visual_aux else None,
                labels_override=visual_aux.get("labels") if visual_aux else None,
            )
        target_frames = ptm_supervision["target_frames"].to(memory_tokens.device)
        target_latents = self.encode(target_frames.unsqueeze(0)).squeeze(0)
        target_embeddings = target_latents.mean(dim=(-2, -1))
        test_loss, components = self.ptm_test_loss(predictions, labels, target_embeddings)
        bottleneck_loss, bottleneck_components = self.ptm_bottleneck_loss(memory_tokens)
        components.update(bottleneck_components)
        components["future_tests_unweighted"] = test_loss
        components["bottleneck_unweighted"] = bottleneck_loss
        return test_loss + bottleneck_loss, components

    def _gather_time_batch(self, tensor: torch.Tensor, time_indices: torch.Tensor | np.ndarray) -> torch.Tensor:
        time_indices = torch.as_tensor(time_indices, dtype=torch.long, device=tensor.device)
        batch_indices = torch.arange(tensor.shape[1], device=tensor.device)
        return tensor[time_indices, batch_indices.unsqueeze(0)]

    def _ptm_sampling_tokens_from_selected_history(
        self,
        xs_pred: torch.Tensor,
        conditions: torch.Tensor,
        pose_conditions: torch.Tensor,
        selected_indices: torch.Tensor | np.ndarray | None,
        start_frame: int,
        curr_frame: int,
        batch: dict | None = None,
    ) -> torch.Tensor | None:
        if not self.use_ptm_memory:
            return None
        if selected_indices is None:
            selected_indices = torch.empty((0, xs_pred.shape[1]), dtype=torch.long, device=xs_pred.device)
        if curr_frame <= start_frame and len(selected_indices) == 0:
            return None
        return self._encode_ptm_memory_input(
            xs_pred,
            conditions,
            pose_conditions,
            selected_indices,
            start_frame,
            curr_frame,
            batch,
        )

    @torch.no_grad()
    def _record_ptm_eval_predictions(
        self,
        batch,
        xs: torch.Tensor,
        conditions: torch.Tensor,
        pose_conditions: torch.Tensor,
        namespace: str,
        batch_idx: int,
        mode: str | None = None,
    ) -> None:
        if not self.use_ptm_memory:
            return
        ptm_supervision = self._extract_ptm_supervision(batch)
        if ptm_supervision is None:
            return
        mode = self._canonical_ablation_mode(mode or self.ptm_ablation)
        original_mode = self.ptm_ablation
        self.ptm_ablation = mode
        selected_indices = self._reference_tail_indices(xs, batch)
        recent_end = self._recent_end_from_batch(xs, batch)
        # For long-history probes (recent window exceeds max_history), the
        # reference-tail prefix gets truncated by PredictiveTestMemory and
        # ablation on it has no effect. Ablate the tail (the recent frames the
        # encoder actually consumes) instead, using max_history as the window.
        recent_len = max(0, recent_end - 0)
        ablate_tail = self.ptm_max_history if recent_len > self.ptm_max_history else 0
        try:
            memory_tokens = self._encode_ptm_memory_input(
                xs,
                conditions,
                pose_conditions,
                selected_indices,
                0,
                recent_end,
                batch,
                ablate_tail=ablate_tail,
            )
            if memory_tokens is None:
                return
            if self.ptm_visual_memory_selection:
                _, visual_aux = self._build_visual_condition_tokens(
                    memory_tokens,
                    xs,
                    batch,
                    ptm_supervision,
                    recent_end,
                )
                predictions = visual_aux["predictions"]
                labels = visual_aux["labels"]
            else:
                predictions, labels = self._ptm_predictions(memory_tokens, ptm_supervision)

            target_frames = ptm_supervision["target_frames"].to(memory_tokens.device)
            target_latents = self.encode(target_frames.unsqueeze(0)).squeeze(0)
            target_embeddings = target_latents.mean(dim=(-2, -1))
            embedding_mse = F.mse_loss(
                predictions["future_embedding"],
                target_embeddings,
                reduction="none",
            ).mean(dim=-1)
            future_test_loss, loss_components = self.ptm_test_loss(predictions, labels, target_embeddings)
        finally:
            self.ptm_ablation = original_mode

        test_type_names = batch.get("test_type", ["unknown"] * memory_tokens.shape[0])
        episode_dirs = batch.get("episode_dir", [""] * memory_tokens.shape[0])
        if isinstance(test_type_names, str):
            test_type_names = [test_type_names] * memory_tokens.shape[0]
        if isinstance(episode_dirs, str):
            episode_dirs = [episode_dirs] * memory_tokens.shape[0]

        outputs = self.ptm_eval_outputs.setdefault(namespace, [])
        loop_prob = torch.sigmoid(predictions["loop_return_logit"]).detach().cpu()
        landmark_prob = torch.sigmoid(predictions["landmark_visible_logit"]).detach().cpu()
        object_prob = torch.sigmoid(predictions["object_exists_logit"]).detach().cpu()
        match_pred = predictions["match_history_logits"].argmax(dim=-1).detach().cpu()
        labels_cpu = {key: value.detach().cpu() for key, value in labels.items()}
        embedding_mse = embedding_mse.detach().cpu()
        batch_loss_payload = {
            "ptm_future_test_loss": float(future_test_loss.detach().cpu().item()),
            **{
                f"ptm_{name}_loss": float(value.detach().cpu().item())
                for name, value in loss_components.items()
            },
        }

        for sample_idx in range(memory_tokens.shape[0]):
            loop_label = float(labels_cpu["returns_to_seen_place"][sample_idx])
            landmark_label = float(labels_cpu["landmark_visible"][sample_idx])
            object_label = float(labels_cpu["object_exists_at_return"][sample_idx])
            loop_score = float(loop_prob[sample_idx])
            landmark_score = float(landmark_prob[sample_idx])
            object_score = float(object_prob[sample_idx])
            outputs.append(
                {
                    "namespace": namespace,
                    "mode": mode,
                    "global_step": self._validation_log_step(namespace) or int(getattr(self, "global_step", 0)),
                    "batch_idx": int(batch_idx),
                    "sample_idx": int(sample_idx),
                    "episode_dir": str(episode_dirs[sample_idx]),
                    "test_type": str(test_type_names[sample_idx]),
                    "test_type_id": int(labels_cpu["test_type_id"][sample_idx]),
                    "returns_to_seen_place": loop_label,
                    "returns_to_seen_place_label": loop_label,
                    "loop_return_prob": loop_score,
                    "returns_to_seen_place_prob": loop_score,
                    "matched_history_index_label": int(labels_cpu["matched_history_index"][sample_idx]),
                    "match_valid": bool(labels_cpu["match_valid"][sample_idx]),
                    "matched_history_index_pred": int(match_pred[sample_idx]),
                    "landmark_visible": landmark_label,
                    "landmark_visible_label": landmark_label,
                    "landmark_visible_prob": landmark_score,
                    "object_exists_at_return": object_label,
                    "object_exists_at_return_label": object_label,
                    "object_exists_prob": object_score,
                    "object_exists_at_return_prob": object_score,
                    "future_embedding_mse": float(embedding_mse[sample_idx]),
                    **batch_loss_payload,
                }
            )

    def _flush_ptm_eval_outputs(self, namespace: str) -> None:
        records = self.ptm_eval_outputs.get(namespace, [])
        if not records:
            return
        trainer = getattr(self, "_trainer", None)
        rank = int(getattr(trainer, "global_rank", 0)) if trainer is not None else 0
        world_size = int(getattr(trainer, "world_size", 1)) if trainer is not None else 1
        shard_index = os.environ.get("PTM_SHARD_INDEX")
        if shard_index is not None:
            rank = int(shard_index)
            world_size = max(world_size, rank + 1)
        rank_suffix = f"_rank{rank}" if world_size > 1 or shard_index is not None else ""
        root = self._validation_artifact_root()
        ptm_eval_root = os.environ.get("PTM_EVAL_OUTPUT_DIR")
        out_dir = Path(ptm_eval_root) / "ptm_eval" if ptm_eval_root else Path(root) / "ptm_eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        log_step = self._validation_log_step(namespace) or int(getattr(self, "global_step", 0))
        by_mode: dict[str, list[dict]] = {}
        for record in records:
            mode = self._canonical_ablation_mode(record.get("mode", self.ptm_ablation))
            by_mode.setdefault(mode, []).append(record)

        for mode, mode_records in by_mode.items():
            out_path = out_dir / f"{namespace}_{mode}_future_test_predictions_step{log_step}{rank_suffix}.jsonl"
            with out_path.open("w", encoding="utf-8") as f:
                for record in mode_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            summary_path = out_dir / f"{namespace}_{mode}_future_test_summary_step{log_step}{rank_suffix}.json"
            summary = self._summarize_ptm_eval_records(mode_records)
            summary["mode"] = mode
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"wrote PTM future-test predictions to {out_path}")
            print(f"wrote PTM future-test summary to {summary_path}")
        self.ptm_eval_outputs[namespace] = []

    def _binary_bce_from_records(self, records: list[dict], prob_key: str, label_key: str, test_type_id: int) -> float | None:
        subset = [record for record in records if int(record.get("test_type_id", -1)) == int(test_type_id)]
        if not subset:
            return None
        probs = torch.tensor([float(record[prob_key]) for record in subset], dtype=torch.float32).clamp(1e-6, 1 - 1e-6)
        labels = torch.tensor([float(record[label_key]) for record in subset], dtype=torch.float32)
        return float(F.binary_cross_entropy(probs, labels).item())

    def _binary_accuracy_from_records(self, records: list[dict], prob_key: str, label_key: str, test_type_id: int) -> float | None:
        subset = [record for record in records if int(record.get("test_type_id", -1)) == int(test_type_id)]
        if not subset:
            return None
        preds = torch.tensor([float(record[prob_key]) >= 0.5 for record in subset], dtype=torch.bool)
        labels = torch.tensor([float(record[label_key]) >= 0.5 for record in subset], dtype=torch.bool)
        return float((preds == labels).float().mean().item())

    def _summarize_ptm_eval_records(self, records: list[dict]) -> dict:
        embedding = torch.tensor([float(record["future_embedding_mse"]) for record in records], dtype=torch.float32)
        valid_match = [record for record in records if bool(record.get("match_valid", False))]
        summary = {
            "num_samples": len(records),
            "future_embedding_mse": float(embedding.mean().item()) if len(records) else None,
            "loop_return_bce": self._binary_bce_from_records(records, "loop_return_prob", "returns_to_seen_place_label", 1),
            "loop_return_accuracy": self._binary_accuracy_from_records(records, "loop_return_prob", "returns_to_seen_place_label", 1),
            "landmark_visible_bce": self._binary_bce_from_records(records, "landmark_visible_prob", "landmark_visible_label", 2),
            "landmark_visible_accuracy": self._binary_accuracy_from_records(records, "landmark_visible_prob", "landmark_visible_label", 2),
            "object_exists_bce": self._binary_bce_from_records(records, "object_exists_prob", "object_exists_at_return_label", 3),
            "object_exists_accuracy": self._binary_accuracy_from_records(records, "object_exists_prob", "object_exists_at_return_label", 3),
            "matched_history_num_valid": len(valid_match),
            "matched_history_accuracy": None,
        }
        if valid_match:
            correct = [
                int(record["matched_history_index_pred"]) == int(record["matched_history_index_label"])
                for record in valid_match
            ]
            summary["matched_history_accuracy"] = float(torch.tensor(correct, dtype=torch.float32).mean().item())
        for key in (
            "ptm_future_test_loss",
            "ptm_future_embedding_loss",
            "ptm_loop_return_loss",
            "ptm_matched_history_loss",
            "ptm_landmark_visible_loss",
            "ptm_object_exists_loss",
        ):
            values = [float(record[key]) for record in records if key in record]
            if values:
                summary[key] = float(torch.tensor(values, dtype=torch.float32).mean().item())
        return summary

    def _batch_values(self, batch: dict | None, key: str, batch_size: int, default):
        if not isinstance(batch, dict) or key not in batch:
            return [default for _ in range(batch_size)]
        value = batch[key]
        if torch.is_tensor(value):
            flat = value.detach().cpu().flatten()
            if flat.numel() == 1 and batch_size > 1:
                flat = flat.expand(batch_size)
            return [flat[i].item() for i in range(min(batch_size, flat.numel()))]
        if isinstance(value, (list, tuple)):
            return list(value)[:batch_size]
        return [value for _ in range(batch_size)]

    def _generation_metadata(
        self,
        batch,
        frame_idx: torch.Tensor,
        n_context_frames: int,
        future_frames: int,
        batch_size: int,
    ) -> list[dict]:
        test_types = self._batch_values(batch, "test_type", batch_size, "unknown")
        episode_dirs = self._batch_values(batch, "episode_dir", batch_size, "")
        episode_families = self._batch_values(batch, "episode_family", batch_size, "")
        labels = batch.get("memory_labels", {}) if isinstance(batch, dict) else {}
        target_times = self._batch_values(batch, "target_t", batch_size, -1)
        generation_centers = self._batch_values(batch, "generation_center_index_in_video", batch_size, -1)
        metadata = []
        frame_idx_cpu = frame_idx.detach().cpu()
        for sample_idx in range(batch_size):
            target_future_index = None
            center_index = int(generation_centers[sample_idx]) if sample_idx < len(generation_centers) else -1
            if center_index >= 0:
                target_future_index = center_index - int(n_context_frames)
            target_abs = int(target_times[sample_idx]) if sample_idx < len(target_times) else -1
            if target_future_index is None and target_abs >= 0:
                matches = (frame_idx_cpu[:, sample_idx] == target_abs).nonzero(as_tuple=False).flatten()
                if matches.numel():
                    target_future_index = int(matches[0].item()) - int(n_context_frames)
            if target_future_index is None:
                target_future_index = max(0, min(future_frames - 1, int(future_frames // 2)))
            sample_labels = {}
            for label_key, label_value in labels.items():
                if torch.is_tensor(label_value):
                    flat = label_value.detach().cpu().flatten()
                    if sample_idx < flat.numel():
                        item = flat[sample_idx].item()
                        sample_labels[label_key] = bool(item) if label_key == "match_valid" else float(item)
                elif isinstance(label_value, (list, tuple)) and sample_idx < len(label_value):
                    sample_labels[label_key] = label_value[sample_idx]
            metadata.append(
                {
                    "sample_idx": sample_idx,
                    "test_type": str(test_types[sample_idx]) if sample_idx < len(test_types) else "unknown",
                    "episode_dir": str(episode_dirs[sample_idx]) if sample_idx < len(episode_dirs) else "",
                    "episode_family": str(episode_families[sample_idx]) if sample_idx < len(episode_families) else "",
                    "target_future_index": int(max(0, min(future_frames - 1, target_future_index))),
                    "labels": sample_labels,
                }
            )
        return metadata

    def _generation_metrics(self, xs_pred: torch.Tensor, xs: torch.Tensor) -> dict[str, float]:
        device = next(self.validation_lpips_model.parameters()).device
        metric_dict = get_validation_metrics_for_videos(
            xs_pred.to(device),
            xs.to(device),
            lpips_model=self.validation_lpips_model,
            lpips_batch_size=self.lpips_batch_size,
        )
        return {
            "mse": float(metric_dict["mse"].detach().cpu().item() if torch.is_tensor(metric_dict["mse"]) else metric_dict["mse"]),
            "psnr": float(metric_dict["psnr"].detach().cpu().item() if torch.is_tensor(metric_dict["psnr"]) else metric_dict["psnr"]),
            "lpips": float(metric_dict["lpips"]),
        }

    def _subset_generation_metrics(
        self,
        xs_pred: torch.Tensor,
        xs: torch.Tensor,
        sample_indices: list[int],
    ) -> dict[str, float] | None:
        if not sample_indices:
            return None
        index = torch.tensor(sample_indices, dtype=torch.long)
        return self._generation_metrics(xs_pred[:, index], xs[:, index])

    def _target_window_generation_metrics(
        self,
        xs_pred: torch.Tensor,
        xs: torch.Tensor,
        metadata: list[dict],
    ) -> dict[str, float] | None:
        if not metadata:
            return None
        radius = max(0, int(self.generation_target_window_radius))
        offsets = list(range(-radius, radius + 1))
        pred_windows = []
        gt_windows = []
        last_frame = xs_pred.shape[0] - 1
        for item in metadata:
            sample_idx = int(item["sample_idx"])
            center = int(item["target_future_index"])
            frame_indices = torch.tensor(
                [max(0, min(last_frame, center + offset)) for offset in offsets],
                dtype=torch.long,
            )
            pred_windows.append(xs_pred[frame_indices, sample_idx:sample_idx + 1])
            gt_windows.append(xs[frame_indices, sample_idx:sample_idx + 1])
        return self._generation_metrics(torch.cat(pred_windows, dim=1), torch.cat(gt_windows, dim=1))

    def _generation_metrics_payload(
        self,
        xs_pred: torch.Tensor,
        xs: torch.Tensor,
        metadata: list[dict],
        namespace: str,
    ) -> dict:
        future_frames = xs_pred.shape[0]
        late_start = min(max(0, int(self.generation_late_horizon_start)), future_frames - 1)
        subsets = {
            "loop_return": [
                item["sample_idx"]
                for item in metadata
                if float(item.get("labels", {}).get("returns_to_seen_place", 0.0)) >= 0.5
            ],
            "landmark_visible": [
                item["sample_idx"]
                for item in metadata
                if float(item.get("labels", {}).get("landmark_visible", 0.0)) >= 0.5
            ],
            "object_exists": [
                item["sample_idx"]
                for item in metadata
                if float(item.get("labels", {}).get("object_exists_at_return", 0.0)) >= 0.5
            ],
        }
        return {
            "namespace": namespace,
            "global_step": self._validation_log_step(namespace) or int(getattr(self, "global_step", 0)),
            "num_samples": xs_pred.shape[1],
            "num_future_frames": future_frames,
            "overall": self._generation_metrics(xs_pred, xs),
            "target_window": self._target_window_generation_metrics(xs_pred, xs, metadata),
            "late_horizon": self._generation_metrics(xs_pred[late_start:], xs[late_start:]),
            "subsets": {
                name: {
                    "num_samples": len(indices),
                    "metrics": self._subset_generation_metrics(xs_pred, xs, indices),
                }
                for name, indices in subsets.items()
            },
        }

    def _write_generation_metrics(
        self,
        namespace: str,
        payload: dict,
        *,
        include_rank_suffix: bool | None = None,
    ) -> None:
        trainer = getattr(self, "_trainer", None)
        rank = int(getattr(trainer, "global_rank", 0)) if trainer is not None else 0
        world_size = int(getattr(trainer, "world_size", 1)) if trainer is not None else 1
        if include_rank_suffix is None:
            include_rank_suffix = world_size > 1
        rank_suffix = f"_rank{rank}" if include_rank_suffix else ""
        root = self._validation_artifact_root()
        out_dir = Path(root) / "generation_eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        log_step = payload["global_step"]
        out_path = out_dir / f"{namespace}_generation_metrics_step{log_step}{rank_suffix}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"wrote generation metrics to {out_path}")

    def _gather_generation_payloads(self, payload: dict) -> list[dict]:
        trainer = getattr(self, "_trainer", None)
        world_size = int(getattr(trainer, "world_size", 1)) if trainer is not None else 1
        if world_size <= 1:
            return [payload]
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError("DDP generation validation requires initialized torch.distributed")
        gathered: list[dict | None] = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(gathered, payload)
        return [item for item in gathered if item is not None]

    def _weighted_generation_metrics(
        self,
        payloads: list[dict],
        scope: str,
        *,
        sample_count_key: str = "num_samples",
    ) -> dict[str, float] | None:
        totals = {"mse": 0.0, "psnr": 0.0, "lpips": 0.0}
        total_samples = 0
        for payload in payloads:
            metrics = payload.get(scope)
            samples = int(payload.get(sample_count_key, 0))
            if not metrics or samples <= 0:
                continue
            for key in totals:
                totals[key] += float(metrics[key]) * samples
            total_samples += samples
        if total_samples <= 0:
            return None
        return {key: value / total_samples for key, value in totals.items()}

    def _aggregate_generation_payloads(self, payloads: list[dict]) -> dict:
        valid_payloads = [payload for payload in payloads if int(payload.get("num_samples", 0)) > 0]
        if not valid_payloads:
            raise ValueError("cannot aggregate empty generation payloads")
        first = valid_payloads[0]
        total_samples = sum(int(payload["num_samples"]) for payload in valid_payloads)
        subset_names = sorted({
            subset_name
            for payload in valid_payloads
            for subset_name in payload.get("subsets", {}).keys()
        })
        subsets = {}
        for subset_name in subset_names:
            subset_payloads = []
            total_subset_samples = 0
            for payload in valid_payloads:
                subset = payload.get("subsets", {}).get(subset_name)
                if not subset:
                    continue
                num_samples = int(subset.get("num_samples", 0))
                total_subset_samples += num_samples
                subset_payloads.append({
                    "num_samples": num_samples,
                    "metrics": subset.get("metrics"),
                })
            metrics = self._weighted_generation_metrics(
                [
                    {"num_samples": item["num_samples"], "metrics": item["metrics"]}
                    for item in subset_payloads
                ],
                "metrics",
            )
            subsets[subset_name] = {
                "num_samples": total_subset_samples,
                "metrics": metrics,
            }
        return {
            "namespace": first["namespace"],
            "global_step": int(first["global_step"]),
            "num_samples": total_samples,
            "num_future_frames": int(first["num_future_frames"]),
            "overall": self._weighted_generation_metrics(valid_payloads, "overall"),
            "target_window": self._weighted_generation_metrics(valid_payloads, "target_window"),
            "late_horizon": self._weighted_generation_metrics(valid_payloads, "late_horizon"),
            "subsets": subsets,
        }

    def _generate_noise_levels(self, xs: torch.Tensor, masks = None) -> torch.Tensor:
        """
        Generate noise levels for training.
        """
        num_frames, batch_size, *_ = xs.shape
        match self.cfg.noise_level:
            case "random_all":  # entirely random noise levels
                noise_levels = torch.randint(0, self.timesteps, (num_frames, batch_size), device=xs.device)
            case "same":
                noise_levels = torch.randint(0, self.timesteps, (num_frames, batch_size), device=xs.device)
                noise_levels[1:] = noise_levels[0]

        if masks is not None:
            # for frames that are not available, treat as full noise
            discard = torch.all(~rearrange(masks.bool(), "(t fs) b -> t b fs", fs=self.frame_stack), -1)
            noise_levels = torch.where(discard, torch.full_like(noise_levels, self.timesteps - 1), noise_levels)

        return noise_levels

    def _required_batch_long_tensor(
        self,
        batch: dict,
        key: str,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if key not in batch:
            raise ValueError(f"batch is missing required field {key!r}")
        value = batch[key]
        if torch.is_tensor(value):
            flat = value.detach().to(device=device, dtype=torch.long).flatten()
        else:
            flat = torch.as_tensor(value, dtype=torch.long, device=device).flatten()
        if flat.numel() != batch_size:
            raise ValueError(f"batch field {key!r} has {flat.numel()} values; expected batch size {batch_size}")
        return flat

    def _required_window_kinds(self, batch: dict, batch_size: int) -> list[str]:
        if "window_kind" not in batch:
            raise ValueError("batch is missing required field 'window_kind'")
        value = batch["window_kind"]
        if isinstance(value, str):
            kinds = [value]
        elif isinstance(value, (list, tuple)):
            kinds = [str(item) for item in value]
        else:
            raise ValueError(f"window_kind must be a string list, got {type(value).__name__}")
        if len(kinds) != batch_size:
            raise ValueError(f"window_kind has {len(kinds)} values; expected batch size {batch_size}")
        return kinds

    def _generation_training_loss_weight(
        self,
        loss: torch.Tensor,
        batch: dict | None,
    ) -> torch.Tensor | None:
        target_gain = float(self.generation_target_loss_weight)
        late_gain = float(self.generation_late_loss_weight)
        if target_gain <= 0.0 and late_gain <= 0.0:
            return None
        if not isinstance(batch, dict):
            raise ValueError("generation target/late loss weighting requires dict batches with window metadata")

        num_tokens, batch_size = loss.shape[:2]
        num_frames = int(num_tokens * self.frame_stack)
        device = loss.device
        centers = self._required_batch_long_tensor(
            batch,
            "generation_center_index_in_video",
            batch_size,
            device,
        )
        kinds = self._required_window_kinds(batch, batch_size)
        radius = max(0, int(self.generation_target_window_radius))
        weight = torch.ones((num_frames, batch_size), dtype=loss.dtype, device=device)
        weighted_any = False

        for sample_idx, kind in enumerate(kinds):
            normalized_kind = kind.strip().lower().replace("-", "_")
            gain = 0.0
            if normalized_kind == "target":
                gain = target_gain
            elif normalized_kind.startswith("late_"):
                gain = late_gain
            if gain <= 0.0:
                continue
            center = int(centers[sample_idx].item())
            if center < 0 or center >= num_frames:
                raise ValueError(
                    f"generation_center_index_in_video={center} is outside training loss length {num_frames}"
                )
            start = max(0, center - radius)
            end = min(num_frames, center + radius + 1)
            weight[start:end, sample_idx] = weight[start:end, sample_idx] + gain
            weighted_any = True

        if not weighted_any:
            return None
        return weight / weight.mean().clamp_min(1e-6)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        """
        Perform a single training step.

        This function processes the input batch,
        encodes the input frames, generates noise levels, and computes the loss using the diffusion model.

        Args:
            batch: Input batch of data containing frames, conditions, poses, etc.
            batch_idx: Index of the current batch.

        Returns:
            dict: A dictionary containing the training loss.
        """
        ptm_supervision = self._extract_ptm_supervision(batch)
        batch_dict = batch if isinstance(batch, dict) else None
        batch_has_ref_tail = bool(self.memory_condition_length) and self._batch_has_reference_tail(batch_dict)
        batch_memory_length = self._batch_int(batch_dict, "memory_condition_length", 0)
        if batch_has_ref_tail and batch_memory_length != int(self.memory_condition_length):
            raise ValueError(
                f"batch memory_condition_length={batch_memory_length}, "
                f"but algorithm memory_condition_length={self.memory_condition_length}"
            )
        tail_length = batch_memory_length if batch_has_ref_tail else 0
        train_reference_length = (
            tail_length
            if batch_has_ref_tail and not self.ptm_train_context_memory_only
            else 0
        )
        xs, conditions, pose_conditions, c2w_mat, frame_idx = self._preprocess_batch(batch)
        main_frame_count = xs.shape[0] - tail_length if batch_has_ref_tail else xs.shape[0]
        if main_frame_count <= 0:
            raise ValueError(f"invalid training main_frame_count={main_frame_count} for xs length {xs.shape[0]}")
        diffusion_input_length = main_frame_count + train_reference_length
        reference_c2w = c2w_mat[main_frame_count:main_frame_count + train_reference_length]
        reference_frame_idx = frame_idx[main_frame_count:main_frame_count + train_reference_length]

        if self.use_plucker:
            if self.relative_embedding:
                input_pose_condition = []
                frame_idx_list = []
                for i in range(main_frame_count):
                    pose_parts = [c2w_mat[i:i + 1]]
                    frame_parts = [frame_idx[i:i + 1] - frame_idx[i:i + 1]]
                    if train_reference_length:
                        pose_parts.append(reference_c2w)
                        frame_parts.append(reference_frame_idx - frame_idx[i:i + 1])
                    input_pose_condition.append(
                        convert_to_plucker(
                            torch.cat(pose_parts).clone(),
                            0,
                            focal_length=self.focal_length,
                            image_height=xs.shape[-2],image_width=xs.shape[-1]
                        ).to(xs.dtype)
                    )
                    frame_idx_list.append(torch.cat(frame_parts).clone())
                input_pose_condition = torch.cat(input_pose_condition)
                frame_idx_list = torch.cat(frame_idx_list)
            else:
                input_pose_condition = convert_to_plucker(
                    c2w_mat[:diffusion_input_length], 0, focal_length=self.focal_length
                ).to(xs.dtype)
                frame_idx_list = frame_idx[:diffusion_input_length]
        else:
            input_pose_condition = pose_conditions[:diffusion_input_length].to(xs.dtype)
            frame_idx_list = None

        xs = self.encode(xs)
        encoded_xs_full = xs
        conditions_full = conditions
        pose_conditions_full = pose_conditions
        ptm_memory_tokens = None
        ptm_visual_aux = None
        recent_end = None
        if self.use_ptm_memory:
            recent_end = self._recent_end_from_batch(encoded_xs_full, batch_dict)
            use_context_token_source = (
                self.ptm_train_context_memory_only
                and (self.ptm_train_context_token_source == "context" or not batch_has_ref_tail)
            )
            if use_context_token_source:
                selected_indices = self._context_memory_indices(
                    recent_end,
                    encoded_xs_full.shape[1],
                    encoded_xs_full.device,
                )
            else:
                selected_indices = self._reference_tail_indices(encoded_xs_full, batch_dict)
            ptm_memory_tokens = self._encode_ptm_memory_input(
                encoded_xs_full,
                conditions_full,
                pose_conditions_full,
                selected_indices,
                0,
                recent_end,
                batch_dict,
                ablation_mode="normal",
            )
        # PTM tokens for DiT consumption. When ptm_detach_for_generation, the
        # DiT consumer is trained by diffusion/contrast loss only; the PTM
        # encoder is trained solely by the future-test loss below on the
        # non-detached tokens.
        ptm_condition_tokens = None
        if ptm_memory_tokens is not None:
            ptm_condition_tokens, ptm_visual_aux = self._build_visual_condition_tokens(
                ptm_memory_tokens,
                encoded_xs_full,
                batch_dict,
                ptm_supervision,
                int(recent_end),
            )
            ptm_condition_tokens = ptm_condition_tokens.detach() if self.ptm_detach_for_generation else ptm_condition_tokens
        xs = encoded_xs_full[:diffusion_input_length]
        conditions = conditions_full[:diffusion_input_length]
        pose_conditions = pose_conditions_full[:diffusion_input_length]
        normal_conditions = conditions.clone()
        xs, conditions, ptm_memory_tokens = self._apply_ptm_memory_references(
            xs,
            normal_conditions,
            pose_conditions,
            ptm_memory_tokens,
        )

        noise_levels = self._generate_noise_levels(xs)

        if train_reference_length:
            noise_levels[-train_reference_length:] = self.diffusion_model.stabilization_level
            conditions[-train_reference_length:] *= 0

        # Shared noise so pos vs neg loss differs only by token identity.
        shared_noise = torch.randn_like(xs)
        shared_noise = torch.clamp(shared_noise, -self.clip_noise, self.clip_noise)

        _, pos_loss = self.diffusion_model(
            xs,
            conditions,
            input_pose_condition,
            noise_levels=noise_levels,
            reference_length=train_reference_length,
            frame_idx=frame_idx_list,
            ptm_memory_tokens=ptm_condition_tokens,
            noise=shared_noise,
        )

        # Reference tail is a condition, not a prediction target.
        pos_main_loss = pos_loss[:-train_reference_length] if train_reference_length else pos_loss

        generation_loss_weight = self._generation_training_loss_weight(
            pos_main_loss,
            batch if isinstance(batch, dict) else None,
        )
        diffusion_loss = self.reweight_loss(pos_main_loss, generation_loss_weight)
        loss = diffusion_loss

        # Same-noise contrast: shuffled tokens should denoise worse. Only
        # pushes the negative branch (positive is detached here), so the main
        # PSNR optimization is not double-pressured.
        contrast_loss = None
        if (
            self.ptm_contrast_weight > 0
            and ptm_condition_tokens is not None
            and ptm_condition_tokens.shape[0] > 1
        ):
            episode_dirs = batch_dict.get("episode_dir") if batch_dict else None
            episode_families = batch_dict.get("episode_family") if batch_dict else None
            order = self._hard_shuffle_order(
                ptm_condition_tokens.shape[0],
                ptm_condition_tokens.device,
                episode_dirs,
                episode_families,
            )
            neg_tokens = ptm_condition_tokens[order]
            _, neg_loss = self.diffusion_model(
                xs,
                conditions,
                input_pose_condition,
                noise_levels=noise_levels,
                reference_length=train_reference_length,
                frame_idx=frame_idx_list,
                ptm_memory_tokens=neg_tokens,
                noise=shared_noise,
            )
            neg_main_loss = neg_loss[:-train_reference_length] if train_reference_length else neg_loss
            neg_scalar = self.reweight_loss(neg_main_loss, generation_loss_weight)
            contrast_loss = torch.relu(self.ptm_contrast_margin + diffusion_loss.detach() - neg_scalar)
            loss = loss + self.ptm_contrast_weight * contrast_loss

        if self.use_ptm_memory and ptm_supervision is not None and self.ptm_loss_weight > 0:
            ptm_loss, ptm_components = self._compute_ptm_test_loss(ptm_memory_tokens, ptm_supervision, ptm_visual_aux)
            loss = loss + self.ptm_loss_weight * ptm_loss
            if batch_idx % 20 == 0:
                self.log("training/ptm_future_test_loss", ptm_components["future_tests_unweighted"].detach().cpu())
                self.log("training/ptm_bottleneck_loss", ptm_components["bottleneck_unweighted"].detach().cpu())

        if batch_idx % 20 == 0:
            self.log("training/diffusion_loss", diffusion_loss.detach().cpu())
            if contrast_loss is not None:
                self.log("training/ptm_contrast_loss", contrast_loss.detach().cpu())
            if ptm_visual_aux is not None:
                self.log(
                    "training/ptm_visual_tokens",
                    torch.as_tensor(ptm_visual_aux["visual_tokens"].shape[1], dtype=torch.float32),
                )
            if generation_loss_weight is not None:
                self.log("training/generation_weight_mean", generation_loss_weight.detach().mean().cpu())
                self.log("training/generation_weight_max", generation_loss_weight.detach().max().cpu())
            self.log("training/loss", loss.detach().cpu())
            if getattr(self.trainer, "global_rank", 0) == 0:
                step = int(getattr(self, "global_step", 0))
                loss_value = float(loss.detach().cpu())
                diffusion_loss_value = float(diffusion_loss.detach().cpu())
                message = (
                    f"[local_loss] step={step} batch_idx={batch_idx} "
                    f"training/loss={loss_value:.8f} "
                    f"training/diffusion_loss={diffusion_loss_value:.8f}"
                )
                print(message, flush=True)
                local_loss_log = os.environ.get("WORLDMEM_LOCAL_LOSS_LOG")
                if local_loss_log:
                    Path(local_loss_log).parent.mkdir(parents=True, exist_ok=True)
                    with open(local_loss_log, "a", encoding="utf-8") as f:
                        f.write(message + "\n")

        return {"loss": loss}
    
    def on_validation_epoch_end(self, namespace="validation") -> None:
        if not self.validation_step_outputs:
            self._flush_ptm_eval_outputs(namespace)
            return

        grouped: dict[str, dict[str, list]] = {}
        for item in self.validation_step_outputs:
            pred, gt = item[:2]
            mode = self._canonical_ablation_mode(item[3] if len(item) > 3 else self.ptm_ablation)
            bucket = grouped.setdefault(mode, {"pred": [], "gt": [], "metadata": []})
            bucket["pred"].append(pred)
            bucket["gt"].append(gt)
            if len(item) > 2 and item[2] is not None:

                bucket["metadata"].extend(item[2])

        compare_payloads = {}
        for mode, bucket in grouped.items():
            xs_pred = torch.cat(bucket["pred"], 1)
            xs = torch.cat(bucket["gt"], 1) if bucket["gt"] and bucket["gt"][0] is not None else None
            metadata = bucket["metadata"]

            if self.logger and self.log_video and mode == self.validation_video_mode:
                log_video(
                    xs_pred,
                    xs,
                    step=self._validation_log_step(namespace),
                    namespace=f"{self._video_log_namespace(namespace)}_{mode}",
                    context_frames=self.context_frames,
                    logger=self.logger.experiment,
                    save_local=self.save_local,
                    local_save_dir=self.local_save_dir,
                    max_videos=self.max_log_videos,
                )

            if xs is None:
                continue

            device = next(self.validation_lpips_model.parameters()).device
            xs_pred_device = xs_pred.to(device)
            xs_device = xs.to(device)

            metric_dict = get_validation_metrics_for_videos(
                xs_pred_device,
                xs_device,
                lpips_model=self.validation_lpips_model,
                lpips_batch_size=self.lpips_batch_size,
            )

            metric_payload = {
                "mse": metric_dict["mse"],
                "psnr": metric_dict["psnr"],
                "lpips": metric_dict["lpips"],
            }
            generation_payload = self._generation_metrics_payload(xs_pred, xs, metadata, f"{namespace}_{mode}")
            self._write_generation_metrics(f"{namespace}_{mode}", generation_payload)
            gathered_payloads = self._gather_generation_payloads(generation_payload)
            generation_payload_all_ranks = self._aggregate_generation_payloads(gathered_payloads)
            if self._can_log_wandb_charts():
                self._write_generation_metrics(
                    f"{namespace}_{mode}_all_ranks",
                    generation_payload_all_ranks,
                    include_rank_suffix=False,
                )
            compare_payloads[mode] = generation_payload_all_ranks["overall"]
            if self.generation_wandb_detailed_metrics:
                log_payload = {
                    f"{namespace}/generation/{mode}/{key}": value
                    for key, value in metric_payload.items()
                }
                for scope in ("target_window", "late_horizon"):
                    scoped_metrics = generation_payload.get(scope)
                    if scoped_metrics is not None:
                        log_payload.update({
                            f"{namespace}/generation/{mode}/{scope}/{key}": value
                            for key, value in scoped_metrics.items()
                        })
                for subset_name, subset_payload in generation_payload["subsets"].items():
                    subset_metrics = subset_payload.get("metrics")
                    if subset_metrics is not None:
                        log_payload.update({
                            f"{namespace}/generation/{mode}/subset_{subset_name}/{key}": value
                            for key, value in subset_metrics.items()
                        })
                self.log_dict(log_payload, sync_dist=True)

            if self.log_curve and mode == self.validation_video_mode and self._can_log_wandb_charts():
                psnr_values = metric_dict['frame_wise_psnr'].cpu().tolist()
                frames = list(range(len(psnr_values)))
                line_plot = wandb.plot.line_series(
                    xs=frames,
                    ys=[psnr_values],
                    keys=[f"PSNR/{mode}"],
                    title=f"Frame-wise PSNR/{mode}",
                    xname="Frame index",
                )
                self.logger.experiment.log({f"frame_wise_psnr_plot/{mode}": line_plot})

        self._log_generation_compare_charts(namespace, compare_payloads)
        self.validation_step_outputs.clear()
        self._flush_ptm_eval_outputs(namespace)

    def _float_metric(self, value) -> float:
        if torch.is_tensor(value):
            return float(value.detach().cpu().item())
        return float(value)

    def _can_log_wandb_charts(self) -> bool:
        trainer = getattr(self, "trainer", None)
        if trainer is not None and hasattr(trainer, "is_global_zero") and not trainer.is_global_zero:
            return False
        if not self.logger:
            return False
        return getattr(self.logger, "experiment", None) is not None

    def _validation_artifact_root(self) -> Path:
        if self.local_save_dir:
            return Path(self.local_save_dir)
        configured_output = getattr(self.cfg, "output_dir", None)
        if configured_output:
            return Path(str(configured_output))
        trainer = getattr(self, "_trainer", None) or getattr(self, "trainer", None)
        default_root = getattr(trainer, "default_root_dir", os.getcwd()) if trainer is not None else os.getcwd()
        return Path(default_root)

    def _log_generation_compare_charts(self, namespace: str, compare_payloads: dict[str, dict[str, float]]) -> None:
        if not compare_payloads or not self._can_log_wandb_charts():
            return
        experiment = self.logger.experiment

        step = int(self._validation_log_step(namespace) or getattr(self, "global_step", 0))
        history = self._generation_compare_history.setdefault(namespace, {})
        for mode in self.validation_ablation_modes:
            metrics = compare_payloads.get(mode)
            if not metrics:
                continue
            mode_history = history.setdefault(mode, {"step": [], "psnr": [], "mse": [], "lpips": []})
            if mode_history["step"] and mode_history["step"][-1] == step:
                for metric_name, metric_value in metrics.items():
                    mode_history[metric_name][-1] = metric_value
            else:
                mode_history["step"].append(step)
                for metric_name in ("psnr", "mse", "lpips"):
                    mode_history[metric_name].append(float(metrics[metric_name]))

        modes = [mode for mode in self.validation_ablation_modes if mode in history]
        if not modes:
            return
        common_steps = sorted(set.intersection(*(set(history[mode]["step"]) for mode in modes)))
        if not common_steps:
            return

        scalars = {}
        for metric_name in ("psnr", "mse", "lpips"):
            for mode in modes:
                step_to_value = dict(zip(history[mode]["step"], history[mode][metric_name]))
                scalars[f"{namespace}/generation_compare/{metric_name}/{mode}"] = float(step_to_value[common_steps[-1]])
        if scalars:
            if hasattr(experiment, "log"):
                experiment.log(scalars, step=step)
            else:
                for key, value in scalars.items():
                    try:
                        experiment.log_metrics({key: value}, step=step)
                    except Exception:
                        pass

    def _preprocess_batch(self, batch):

        if isinstance(batch, dict):
            xs = batch["video"]
            conditions = batch["actions"]
            pose_conditions = batch["poses"]
            frame_index = batch["timestamp"]
        else:
            xs, conditions, pose_conditions, frame_index = batch

        if self.action_cond_dim:
            conditions = torch.cat([torch.zeros_like(conditions[:, :1]), conditions[:, 1:]], 1)
            conditions = rearrange(conditions, "b t d -> t b d").contiguous()
        else:
            raise NotImplementedError("Only support external cond.")

        pose_conditions = rearrange(pose_conditions, "b t d -> t b d").contiguous()
        c2w_mat = euler_to_camera_to_world_matrix(pose_conditions)
        xs = rearrange(xs, "b t c ... -> t b c ...").contiguous()
        frame_index = rearrange(frame_index, "b t -> t b").contiguous()

        return xs, conditions, pose_conditions, c2w_mat, frame_index
    
    def encode(self, x):
        # vae encoding
        T = x.shape[0]
        H, W = x.shape[-2:]
        scaling_factor = 0.07843137255

        x = rearrange(x, "t b c h w -> (t b) c h w")
        with torch.no_grad():
            x = self.vae.encode(x * 2 - 1).mean * scaling_factor
        x = rearrange(x, "(t b) (h w) c -> t b c h w", t=T, h=H // self.vae.patch_size, w=W // self.vae.patch_size)
        return x

    def decode(self, x):
        total_frames = x.shape[0]
        scaling_factor = 0.07843137255
        x = rearrange(x, "t b c h w -> (t b) (h w) c")
        with torch.no_grad():
            x = (self.vae.decode(x / scaling_factor) + 1) / 2
        x = rearrange(x, "(t b) c h w-> t b c h w", t=total_frames)
        return x

    def _generate_condition_indices(self, curr_frame, memory_condition_length, xs_pred, pose_conditions, frame_idx, horizon):
        """
        Generate indices for condition similarity based on the current frame and pose conditions.
        """
        if curr_frame < memory_condition_length:
            random_idx = [i for i in range(curr_frame)] + [0] * (memory_condition_length - curr_frame)
            random_idx = np.repeat(np.array(random_idx)[:, None], xs_pred.shape[1], -1)
        else:
            # Generate points in a sphere and filter based on field of view
            num_samples = 10000
            radius = 30
            points = generate_points_in_sphere(num_samples, radius).to(pose_conditions.device)
            points = points[:, None].repeat(1, pose_conditions.shape[1], 1)
            points += pose_conditions[curr_frame, :, :3][None]
            fov_half_h = torch.tensor(105 / 2, device=pose_conditions.device)
            fov_half_v = torch.tensor(75 / 2, device=pose_conditions.device)

            # in_fov1 = is_inside_fov_3d_hv(
            #     points, pose_conditions[curr_frame, :, :3],
            #     pose_conditions[curr_frame, :, -2], pose_conditions[curr_frame, :, -1],
            #     fov_half_h, fov_half_v
            # )

            in_fov1 = torch.stack([
                is_inside_fov_3d_hv(points, pc[:, :3], pc[:, -2], pc[:, -1], fov_half_h, fov_half_v)
                for pc in pose_conditions[curr_frame:curr_frame+horizon]
            ])

            in_fov1 = torch.sum(in_fov1, 0) > 0

            # Compute overlap ratios and select indices
            in_fov_list = torch.stack([
                is_inside_fov_3d_hv(points, pc[:, :3], pc[:, -2], pc[:, -1], fov_half_h, fov_half_v)
                for pc in pose_conditions[:curr_frame]
            ])

            random_idx = []
            for _ in range(memory_condition_length):
                overlap_ratio = ((in_fov1.bool() & in_fov_list).sum(1)) / in_fov1.sum()
                
                confidence = overlap_ratio + (curr_frame - frame_idx[:curr_frame]) / curr_frame * (-0.2)

                if len(random_idx) > 0:
                    confidence[torch.cat(random_idx)] = -1e10
                _, r_idx = torch.topk(confidence, k=1, dim=0)
                random_idx.append(r_idx[0])

                # choice 1: directly remove overlapping region
                occupied_mask = in_fov_list[r_idx[0, range(in_fov1.shape[-1])], :, range(in_fov1.shape[-1])].permute(1,0)
                in_fov1 = in_fov1 & ~occupied_mask

                # choice 2: apply similarity filter 
                # cos_sim = F.cosine_similarity(xs_pred.to(r_idx.device)[r_idx[:, range(in_fov1.shape[1])], 
                #     range(in_fov1.shape[1])], xs_pred.to(r_idx.device)[:curr_frame], dim=2)
                # cos_sim = cos_sim.mean((-2,-1))

                # mask_sim = cos_sim>0.9
                # in_fov_list = in_fov_list & ~mask_sim[:,None].to(in_fov_list.device)

            random_idx = torch.stack(random_idx).cpu()

        return random_idx

    def _prepare_conditions(self, 
                            start_frame, curr_frame, horizon, conditions, 
                            pose_conditions, c2w_mat, frame_idx, random_idx,
                            image_width, image_height):
        """
        Prepare input conditions and pose conditions for sampling.
        """

        padding = torch.zeros((len(random_idx),) + conditions.shape[1:], device=conditions.device, dtype=conditions.dtype)
        input_condition = torch.cat([conditions[start_frame:curr_frame + horizon], padding], dim=0)

        batch_size = conditions.shape[1]

        if self.use_plucker:
            if self.relative_embedding:
                frame_idx_list = []
                input_pose_condition = []
                for i in range(start_frame, curr_frame + horizon):
                    input_pose_condition.append(convert_to_plucker(torch.cat([c2w_mat[i:i+1],c2w_mat[random_idx[:,range(batch_size)], range(batch_size)]]).clone(), 0, focal_length=self.focal_length,
                                                image_width=image_width, image_height=image_height).to(conditions.dtype))
                    frame_idx_list.append(torch.cat([frame_idx[i:i+1]-frame_idx[i:i+1], frame_idx[random_idx[:,range(batch_size)], range(batch_size)]-frame_idx[i:i+1]]))
                input_pose_condition = torch.cat(input_pose_condition)
                frame_idx_list = torch.cat(frame_idx_list)

            else:
                input_pose_condition = torch.cat([c2w_mat[start_frame : curr_frame + horizon], c2w_mat[random_idx[:,range(batch_size)], range(batch_size)]], dim=0).clone()
                input_pose_condition = convert_to_plucker(input_pose_condition, 0, focal_length=self.focal_length)
                frame_idx_list = None
        else:
            input_pose_condition = torch.cat([pose_conditions[start_frame : curr_frame + horizon], pose_conditions[random_idx[:,range(batch_size)], range(batch_size)]], dim=0).clone()
            frame_idx_list = None

        return input_condition, input_pose_condition, frame_idx_list

    def _prepare_noise_levels(self, scheduling_matrix, m, curr_frame, batch_size, memory_condition_length):
        """
        Prepare noise levels for the current sampling step.
        """
        from_noise_levels = np.concatenate((np.zeros((curr_frame,), dtype=np.int64), scheduling_matrix[m]))[:, None].repeat(batch_size, axis=1)
        to_noise_levels = np.concatenate((np.zeros((curr_frame,), dtype=np.int64), scheduling_matrix[m + 1]))[:, None].repeat(batch_size, axis=1)
        if memory_condition_length:
            from_noise_levels = np.concatenate([from_noise_levels, np.zeros((memory_condition_length, from_noise_levels.shape[-1]), dtype=np.int32)], axis=0)
            to_noise_levels = np.concatenate([to_noise_levels, np.zeros((memory_condition_length, from_noise_levels.shape[-1]), dtype=np.int32)], axis=0)
        from_noise_levels = torch.from_numpy(from_noise_levels).to(self.device)
        to_noise_levels = torch.from_numpy(to_noise_levels).to(self.device)
        return from_noise_levels, to_noise_levels

    def validation_step(self, batch, batch_idx, namespace="validation") -> STEP_OUTPUT:
        """
        Perform a single validation step.

        This function processes the input batch, encodes frames, generates predictions using a sliding window approach,
        and handles condition similarity logic for sampling. The results are decoded and stored for evaluation.

        Args:
            batch: Input batch of data containing frames, conditions, poses, etc.
            batch_idx: Index of the current batch.
            namespace: Namespace for logging (default: "validation").

        Returns:
            None: Appends the predicted and ground truth frames to `self.validation_step_outputs`.
        """
        # Preprocess the input batch
        # has_reference_tail comes from batch (does the data actually have tail frames).
        # memory_condition_length comes from config/CLI (algorithm controls how many tail frames to use).
        sample_has_reference_tail = self._batch_has_reference_tail(batch if isinstance(batch, dict) else None)
        batch_memory_length = self._batch_int(
            batch if isinstance(batch, dict) else None, "memory_condition_length", 0
        )
        memory_condition_length = int(self.memory_condition_length)
        if sample_has_reference_tail:
            if batch_memory_length != memory_condition_length:
                raise ValueError(
                    f"batch memory_condition_length={batch_memory_length}, "
                    f"but algorithm memory_condition_length={memory_condition_length}"
                )
        use_reference_tail = sample_has_reference_tail and not self.ptm_context_memory_only
        tail_length = batch_memory_length if sample_has_reference_tail else 0
        xs_raw, conditions, pose_conditions, c2w_mat, frame_idx = self._preprocess_batch(batch)


        # Encode frames in chunks if necessary
        total_frame = xs_raw.shape[0]
        if total_frame > 10:
            xs = torch.cat([
                self.encode(xs_raw[int(total_frame * i / 10):int(total_frame * (i + 1) / 10)]).cpu()
                for i in range(10)
            ])
        else:
            xs = self.encode(xs_raw).cpu()

        ptm_supervision = self._extract_ptm_supervision(batch)
        original_mode = self.ptm_ablation
        try:
            if self.ptm_eval_only:
                if self.ptm_context_memory_only:
                    raise ValueError("ptm_eval_only is not supported with ptm_context_memory_only")
                for mode in self.validation_ablation_modes:
                    self.ptm_ablation = mode
                    self._record_ptm_eval_predictions(batch, xs, conditions, pose_conditions, namespace, batch_idx, mode)
                return

            if self.use_ptm_memory and not self.ptm_context_memory_only:
                for mode in self.validation_ablation_modes:
                    self._record_ptm_eval_predictions(batch, xs, conditions, pose_conditions, namespace, batch_idx, mode)

            # For eval, always use config/CLI values, ignore batch's context/future length
            main_frames = xs.shape[0] - tail_length if sample_has_reference_tail else xs.shape[0]
            if main_frames <= 0 or main_frames > xs.shape[0]:
                raise ValueError(f"invalid validation main frame count {main_frames} for encoded length {xs.shape[0]}")
            if sample_has_reference_tail and xs.shape[0] < main_frames + tail_length:
                raise ValueError(
                    f"validation batch declares reference tail length {tail_length}, "
                    f"but encoded length={xs.shape[0]} and main_frames={main_frames}"
                )

            xs_main = xs[:main_frames]
            conditions_main = conditions[:main_frames]
            pose_conditions_main = pose_conditions[:main_frames]
            c2w_mat_main = c2w_mat[:main_frames]
            frame_idx_main = frame_idx[:main_frames]
            if use_reference_tail and memory_condition_length:
                ref_start = main_frames
                ref_end = main_frames + memory_condition_length
                reference_xs = xs[ref_start:ref_end]
                reference_actions = conditions[ref_start:ref_end]
                reference_poses = pose_conditions[ref_start:ref_end]
                condition_source = conditions
                pose_source = pose_conditions
                c2w_source = c2w_mat
                frame_idx_source = frame_idx
                reference_indices = torch.arange(
                    ref_start,
                    ref_end,
                    dtype=torch.long,
                    device=frame_idx.device,
                )[:, None].expand(-1, xs.shape[1])
            else:
                reference_xs = reference_actions = reference_poses = None
                condition_source = conditions_main
                pose_source = pose_conditions_main
                c2w_source = c2w_mat_main
                frame_idx_source = frame_idx_main
                reference_indices = None

            n_frames, batch_size, *_ = xs_main.shape
            n_context_frames = self.context_frames // self.frame_stack
            if n_context_frames <= 0 or n_context_frames >= n_frames:
                raise ValueError(f"invalid validation context frames {n_context_frames} for main length {n_frames}")

            for mode in self.validation_ablation_modes:
                self.ptm_ablation = mode
                curr_frame = 0
                xs_pred = xs_main[:n_context_frames].clone()
                curr_frame += n_context_frames
                pbar = tqdm(total=n_frames, initial=curr_frame, desc=f"Sampling/{mode}")
                context_ptm_tokens = None
                if self.ptm_context_memory_only and self.use_ptm_memory and not self.ptm_visual_memory_selection:
                    context_selected = self._context_memory_indices(
                        n_context_frames,
                        batch_size,
                        xs_main.device,
                    )
                    context_ptm_tokens = self._encode_ptm_memory_input(
                        xs_main,
                        conditions_main,
                        pose_conditions_main,
                        context_selected,
                        0,
                        n_context_frames,
                        batch,
                    )

                while curr_frame < n_frames:
                    horizon = min(n_frames - curr_frame, self.chunk_size) if self.chunk_size > 0 else n_frames - curr_frame
                    assert horizon <= self.n_tokens, "Horizon exceeds the number of tokens."

                    scheduling_matrix = self._generate_scheduling_matrix(horizon)
                    chunk = torch.randn((horizon, batch_size, *xs_pred.shape[2:]))
                    chunk = torch.clamp(chunk, -self.clip_noise, self.clip_noise).to(xs_pred.device)
                    xs_pred = torch.cat([xs_pred, chunk], 0)

                    start_frame = max(0, curr_frame + horizon - self.n_tokens)
                    pbar.set_postfix({"start": start_frame, "end": curr_frame + horizon})

                    random_idx = np.zeros((0, batch_size), dtype=np.int64)
                    ptm_sampling_tokens = None
                    ptm_visual_source_xs = None
                    ptm_visual_recent_end = None
                    effective_reference_length = 0
                    if memory_condition_length:
                        if use_reference_tail:
                            random_idx = reference_indices
                            xs_pred = torch.cat([xs_pred, reference_xs.clone()], 0)
                            effective_reference_length = memory_condition_length
                            ptm_source_xs = torch.cat([reference_xs, xs_pred[:-memory_condition_length]], dim=0)
                            ptm_source_actions = torch.cat([reference_actions, conditions_main], dim=0)
                            ptm_source_poses = torch.cat([reference_poses, pose_conditions_main], dim=0)
                            selected_indices = torch.arange(
                                memory_condition_length,
                                dtype=torch.long,
                                device=xs_pred.device,
                            )[:, None].expand(-1, batch_size)
                            if self.use_ptm_memory:
                                ptm_sampling_tokens = self._encode_ptm_memory_input(
                                    ptm_source_xs,
                                    ptm_source_actions,
                                    ptm_source_poses,
                                    selected_indices,
                                    memory_condition_length + start_frame,
                                    memory_condition_length + curr_frame,
                                    batch,
                                )
                                ptm_visual_source_xs = ptm_source_xs
                                ptm_visual_recent_end = memory_condition_length + curr_frame
                        elif self.ptm_context_memory_only:
                            if self.use_ptm_memory and self.ptm_visual_memory_selection:
                                context_selected = self._context_memory_indices(
                                    curr_frame,
                                    batch_size,
                                    xs_pred.device,
                                )
                                ptm_sampling_tokens = self._encode_ptm_memory_input(
                                    xs_pred,
                                    conditions_main,
                                    pose_conditions_main,
                                    context_selected,
                                    0,
                                    curr_frame,
                                    batch,
                                )
                                ptm_visual_source_xs = xs_pred
                                ptm_visual_recent_end = curr_frame
                            else:
                                ptm_sampling_tokens = context_ptm_tokens
                        else:
                            random_idx = self._generate_condition_indices(
                                curr_frame,
                                memory_condition_length,
                                xs_pred,
                                pose_conditions_main,
                                frame_idx_main,
                                horizon,
                            )
                            xs_pred = torch.cat([
                                xs_pred,
                                xs_pred[random_idx[:, range(batch_size)], range(batch_size)].clone(),
                            ], 0)
                            effective_reference_length = memory_condition_length
                            ptm_sampling_tokens = self._ptm_sampling_tokens_from_selected_history(
                                xs_pred,
                                conditions_main,
                                pose_conditions_main,
                                random_idx,
                                start_frame,
                                curr_frame,
                                batch,
                            )
                            ptm_visual_source_xs = xs_pred
                            ptm_visual_recent_end = curr_frame

                    input_condition, input_pose_condition, frame_idx_list = self._prepare_conditions(
                        start_frame,
                        curr_frame,
                        horizon,
                        condition_source,
                        pose_source,
                        c2w_source,
                        frame_idx_source,
                        random_idx,
                        image_width=xs_raw.shape[-1],
                        image_height=xs_raw.shape[-2],
                    )
                    if (
                        effective_reference_length
                        and self.use_ptm_reference_adapter
                        and ptm_sampling_tokens is not None
                    ):
                        ref_latents = self._ptm_reference_latents(
                            ptm_sampling_tokens,
                            effective_reference_length,
                            xs_pred.dtype,
                            xs_pred.shape[-2:],
                        )
                        xs_pred[-effective_reference_length:] = ref_latents.detach().cpu()

                    if (
                        self.ptm_visual_memory_selection
                        and ptm_sampling_tokens is not None
                        and ptm_visual_source_xs is not None
                        and ptm_visual_recent_end is not None
                    ):
                        ptm_sampling_tokens, _ = self._build_visual_condition_tokens(
                            ptm_sampling_tokens,
                            ptm_visual_source_xs,
                            batch,
                            ptm_supervision,
                            int(ptm_visual_recent_end),
                        )

                    # Kill-switch 1: token-level ablation on ptm_sampling_tokens.
                    token_mode = self._canonical_ablation_mode(self.ptm_ablation)
                    if ptm_sampling_tokens is not None and token_mode == "zero_token":
                        ptm_sampling_tokens = torch.zeros_like(ptm_sampling_tokens)
                    elif ptm_sampling_tokens is not None and token_mode == "shuffle_token":
                        ptm_sampling_tokens = self._validation_shuffle_tokens_by_episode(ptm_sampling_tokens, batch)

                    for m in range(scheduling_matrix.shape[0] - 1):
                        from_noise_levels, to_noise_levels = self._prepare_noise_levels(
                            scheduling_matrix, m, curr_frame, batch_size, effective_reference_length
                        )

                        xs_pred[start_frame:] = self.diffusion_model.sample_step(
                            xs_pred[start_frame:].to(input_condition.device),
                            input_condition,
                            input_pose_condition,
                            from_noise_levels[start_frame:],
                            to_noise_levels[start_frame:],
                            current_frame=curr_frame,
                            mode="validation",
                            reference_length=effective_reference_length,
                            frame_idx=frame_idx_list,
                            ptm_memory_tokens=ptm_sampling_tokens,
                        ).cpu()

                    if effective_reference_length:
                        xs_pred = xs_pred[:-effective_reference_length]

                    curr_frame += horizon
                    pbar.update(horizon)

                xs_pred_decoded = self.decode(xs_pred[n_context_frames:].to(conditions.device))
                xs_decoded = self.decode(xs_main[n_context_frames:].to(conditions.device))
                metadata = self._generation_metadata(
                    batch,
                    frame_idx_main,
                    n_context_frames,
                    xs_pred_decoded.shape[0],
                    xs_pred_decoded.shape[1],
                )
                self.validation_step_outputs.append(
                    (xs_pred_decoded.detach().cpu(), xs_decoded.detach().cpu(), metadata, mode)
                )
        finally:
            self.ptm_ablation = original_mode
        return

    @torch.no_grad()
    def interactive(self, first_frame, new_actions, first_pose, device,
                    memory_latent_frames, memory_actions, memory_poses, memory_c2w, memory_frame_idx):
    
        memory_condition_length = self.memory_condition_length

        if memory_latent_frames is None:
            first_frame = torch.from_numpy(first_frame)
            new_actions = torch.from_numpy(new_actions)
            first_pose = torch.from_numpy(first_pose)
            first_frame_encode = self.encode(first_frame[None, None].to(device))
            memory_latent_frames = first_frame_encode.cpu()
            memory_actions = new_actions[None, None].to(device)
            memory_poses = first_pose[None, None].to(device)
            new_c2w_mat = euler_to_camera_to_world_matrix(first_pose)
            memory_c2w = new_c2w_mat[None, None].to(device)
            memory_frame_idx = torch.tensor([[0]]).to(device)
            return first_frame.cpu().numpy(), memory_latent_frames.cpu().numpy(), memory_actions.cpu().numpy(), memory_poses.cpu().numpy(), memory_c2w.cpu().numpy(), memory_frame_idx.cpu().numpy()
        else:
            memory_latent_frames = torch.from_numpy(memory_latent_frames)
            memory_actions = torch.from_numpy(memory_actions).to(device)
            memory_poses = torch.from_numpy(memory_poses).to(device)
            memory_c2w = torch.from_numpy(memory_c2w).to(device)
            memory_frame_idx = torch.from_numpy(memory_frame_idx).to(device)
            new_actions = new_actions.to(device)

        curr_frame = 0
        batch_size = 1
        horizon = self.next_frame_length
        n_frames = curr_frame + horizon
        # context
        n_context_frames = len(memory_latent_frames)
        xs_pred = memory_latent_frames[:n_context_frames].clone()
        curr_frame += n_context_frames

        pbar = tqdm(total=n_frames, initial=curr_frame, desc="Sampling")

        new_pose_condition_list = []
        last_frame = xs_pred[-1].clone()
        last_pose_condition = memory_poses[-1].clone()
        curr_actions = new_actions.clone()
        for hi in range(len(new_actions)):
            last_pose_condition[:,3:] = last_pose_condition[:,3:] // 15
            new_pose_condition_offset = self.pose_prediction_model(last_frame.to(device), curr_actions[None, hi], last_pose_condition)
            new_pose_condition_offset[:,3:] = torch.round(new_pose_condition_offset[:,3:])
            new_pose_condition = last_pose_condition + new_pose_condition_offset
            new_pose_condition[:,3:] = new_pose_condition[:,3:] * 15
            new_pose_condition[:,3:] %= 360
            last_pose_condition = new_pose_condition.clone()
            new_pose_condition_list.append(new_pose_condition[None])
        new_pose_condition_list = torch.cat(new_pose_condition_list, 0)
        
        ai = 0
        while ai < len(new_actions):
            next_horizon = min(horizon, len(new_actions) - ai)
            last_frame = xs_pred[-1].clone()
            curr_actions = new_actions[ai:ai+next_horizon].clone()

            new_pose_condition = new_pose_condition_list[ai:ai+next_horizon].clone()

            new_c2w_mat = euler_to_camera_to_world_matrix(new_pose_condition)
            memory_poses = torch.cat([memory_poses, new_pose_condition])
            memory_actions = torch.cat([memory_actions, curr_actions[:, None]])
            memory_c2w = torch.cat([memory_c2w, new_c2w_mat])
            new_indices = memory_frame_idx[-1,0] + torch.arange(next_horizon, device=memory_frame_idx.device) + 1

            memory_frame_idx = torch.cat([memory_frame_idx, new_indices[:, None]])

            conditions = memory_actions.clone()
            pose_conditions = memory_poses.clone()
            c2w_mat = memory_c2w .clone()
            frame_idx = memory_frame_idx.clone()

            # generation on frame
            scheduling_matrix = self._generate_scheduling_matrix(next_horizon)
            chunk = torch.randn((next_horizon, batch_size, *xs_pred.shape[2:])).to(xs_pred.device)
            chunk = torch.clamp(chunk, -self.clip_noise, self.clip_noise)

            xs_pred = torch.cat([xs_pred, chunk], 0)

            # sliding window: only input the last n_tokens frames
            start_frame = max(0, curr_frame - self.n_tokens)

            pbar.set_postfix(
                {
                    "start": start_frame,
                    "end": curr_frame + next_horizon,
                }
            )

            # Handle condition similarity logic
            random_idx = np.zeros((0, xs_pred.shape[1]), dtype=np.int64)
            if memory_condition_length:
                random_idx = self._generate_condition_indices(
                    curr_frame, memory_condition_length, xs_pred, pose_conditions, frame_idx, next_horizon
                )
                
                # random_idx = np.unique(random_idx)[:, None]
                # memory_condition_length = len(random_idx)
                xs_pred = torch.cat([xs_pred, xs_pred[random_idx[:, range(xs_pred.shape[1])], range(xs_pred.shape[1])].clone()], 0)

            # Prepare input conditions and pose conditions
            input_condition, input_pose_condition, frame_idx_list = self._prepare_conditions(
                start_frame, curr_frame, next_horizon, conditions, pose_conditions, c2w_mat, frame_idx, random_idx,
                image_width=first_frame.shape[-1], image_height=first_frame.shape[-2]
            )
            ptm_sampling_tokens = self._ptm_sampling_tokens_from_selected_history(
                xs_pred,
                conditions,
                pose_conditions,
                random_idx,
                start_frame,
                curr_frame,
                None,
            )

            # Perform sampling for each step in the scheduling matrix
            for m in range(scheduling_matrix.shape[0] - 1):
                from_noise_levels, to_noise_levels = self._prepare_noise_levels(
                    scheduling_matrix, m, curr_frame, batch_size, memory_condition_length
                )

                xs_pred[start_frame:] = self.diffusion_model.sample_step(
                    xs_pred[start_frame:].to(input_condition.device),
                    input_condition,
                    input_pose_condition,
                    from_noise_levels[start_frame:],
                    to_noise_levels[start_frame:],
                    current_frame=curr_frame,
                    mode="validation",
                    reference_length=memory_condition_length,
                    frame_idx=frame_idx_list,
                    ptm_memory_tokens=ptm_sampling_tokens,
                ).cpu()


            if memory_condition_length:
                xs_pred = xs_pred[:-memory_condition_length]

            curr_frame += next_horizon
            pbar.update(next_horizon)
            ai += next_horizon

        memory_latent_frames = torch.cat([memory_latent_frames, xs_pred[n_context_frames:]])
        xs_pred = self.decode(xs_pred[n_context_frames:].to(device)).cpu()

        return xs_pred.cpu().numpy(), memory_latent_frames.cpu().numpy(), memory_actions.cpu().numpy(), \
            memory_poses.cpu().numpy(), memory_c2w.cpu().numpy(), memory_frame_idx.cpu().numpy()
