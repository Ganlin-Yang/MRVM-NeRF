"""
Main model implementation
"""
import torch.nn as nn
import torch
from .encoder import ImageEncoder
from .code import PositionalEncoding
from .model_util import make_encoder, make_mlp
import torch.autograd.profiler as profiler
from util import repeat_interleave
import os
import os.path as osp
import warnings
import copy
def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

def update_moving_average(gamma, target_model, online_model):
    for old_params, new_params in zip(target_model.parameters(), online_model.parameters()):
        old_weight, new_weight = old_params.data, new_params.data
        old_params.data = old_weight * gamma + (1. - gamma) * new_weight

def MLP(in_dim, out_dim, hidden_dim):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim)
    )
def produce_corresponding_target(given_online):
    target = copy.deepcopy(given_online)
    set_requires_grad(target, False)
    return target
class PixelNeRFNet(torch.nn.Module):
    def __init__(self, conf, stop_encoder_grad=False):
        """
        :param conf PyHocon config subtree 'model'
        """
        super().__init__()
        self.encoder = make_encoder(conf["encoder"])
        self.use_encoder = conf.get_bool("use_encoder", True)  # Image features?

        self.use_xyz = conf.get_bool("use_xyz", False)

        assert self.use_encoder or self.use_xyz  # Must use some feature..

        self.normalize_z = conf.get_bool("normalize_z", True)
        self.stop_encoder_grad = stop_encoder_grad  # Stop ConvNet gradient (freeze weights)

        self.use_code = conf.get_bool("use_code", False)  # Positional encoding
        self.use_code_viewdirs = conf.get_bool(
            "use_code_viewdirs", True
        )  # Positional encoding applies to viewdirs

        # Enable view directions
        self.use_viewdirs = conf.get_bool("use_viewdirs", False)

        # Global image features?
        self.use_global_encoder = conf.get_bool("use_global_encoder", False)

        d_latent = self.encoder.latent_size if self.use_encoder else 0
        # d_in = 3 if self.use_xyz else 1
        d_in = 0
        if self.use_code or self.use_code_viewdirs:
            self.code = PositionalEncoding.from_conf(conf["code"], d_in=3)
            freq_num = self.code.d_out
        if self.use_code:
            d_in += freq_num
        if self.use_viewdirs:
            if self.use_code_viewdirs:
               d_in += freq_num
            else:
               d_in += 3

        if self.use_global_encoder:
            # Global image feature
            self.global_encoder = ImageEncoder.from_conf(conf["global_encoder"])
            self.global_latent_size = self.global_encoder.latent_size
            d_latent += self.global_latent_size

        d_out = 4

        self.latent_size = self.encoder.latent_size
        self.stage = conf["stage"]
        self.hidden_size = conf["mlp_fine"]['nerformer']['hidden_size']
        self.mask_ratio = conf["mask_ratio_on_a_ray"]
        self.mlp_coarse = make_mlp(conf["mlp_coarse"], d_in=d_in, d_latent=d_latent, d_out=d_out)
        self.mlp_fine = make_mlp(conf["mlp_fine"], d_in=d_in, d_latent=d_latent, d_out=d_out)
        # Define BYOL architecture
        self.moving_average_decay = conf['moving_avg_param']
        self.projection_size = conf['projection_size']
        self.predictor_hidden_size = conf['predictor_hidden_size']
        self.projector = MLP(self.hidden_size, self.projection_size, self.hidden_size)
        self.target_projector = produce_corresponding_target(self.projector)
        self.predictor = MLP(self.projection_size, self.projection_size, self.predictor_hidden_size)
        # Note: this is world -> camera, and bottom row is omitted
        self.register_buffer("poses", torch.empty(1, 3, 4), persistent=False)
        self.register_buffer("image_shape", torch.empty(2), persistent=False)

        self.d_in = d_in
        self.d_out = d_out
        self.d_latent = d_latent
        self.register_buffer("focal", torch.empty(1, 2), persistent=False)
        # Principal point
        self.register_buffer("c", torch.empty(1, 2), persistent=False)
        self.coarse_num = conf.get_int('coarse_pt_num')
        self.fine_num = conf.get_int('fine_pt_num')
        self.num_objs = 0
        self.num_views_per_obj = 1

    def encode(self, images, poses, focal, z_bounds=None, c=None):
        """
        :param images (NS, 3, H, W)
        NS is number of input (aka source or reference) views
        :param poses (NS, 4, 4)
        :param focal focal length () or (2) or (NS) or (NS, 2) [fx, fy]
        :param z_bounds ignored argument (used in the past)
        :param c principal point None or () or (2) or (NS) or (NS, 2) [cx, cy],
        default is center of image
        """
        self.num_objs = images.size(0)
        if len(images.shape) == 5:
            assert len(poses.shape) == 4
            assert poses.size(1) == images.size(
                1
            )  # Be consistent with NS = num input views
            self.num_views_per_obj = images.size(1)
            images = images.reshape(-1, *images.shape[2:])
            poses = poses.reshape(-1, 4, 4)
        else:
            self.num_views_per_obj = 1

        self.encoder(images)
        self.camera_location = poses[:, :3, 3:] # (SB*NS, 3, 1)
        rot = poses[:, :3, :3].transpose(1, 2)  # (B, 3, 3)
        trans = -torch.bmm(rot, poses[:, :3, 3:])  # (B, 3, 1)
        self.poses = torch.cat((rot, trans), dim=-1)  # (B, 3, 4)

        self.image_shape[0] = images.shape[-1]
        self.image_shape[1] = images.shape[-2]

        # Handle various focal length/principal point formats
        if len(focal.shape) == 0:
            # Scalar: fx = fy = value for all views
            focal = focal[None, None].repeat((1, 2))
        elif len(focal.shape) == 1:
            # Vector f: fx = fy = f_i *for view i*
            # Length should match NS (or 1 for broadcast)
            focal = focal.unsqueeze(-1).repeat((1, 2))
        else:
            focal = focal.clone()
        self.focal = focal.float()
        self.focal[..., 1] *= -1.0

        if c is None:
            # Default principal point is center of image
            c = (self.image_shape * 0.5).unsqueeze(0)
        elif len(c.shape) == 0:
            # Scalar: cx = cy = value for all views
            c = c[None, None].repeat((1, 2))
        elif len(c.shape) == 1:
            # Vector c: cx = cy = c_i *for view i*
            c = c.unsqueeze(-1).repeat((1, 2))
        self.c = c

        if self.use_global_encoder:
            self.global_encoder(images)

    def forward(self, xyz, coarse=True, viewdirs=None, far=False):
        """
        Predict (r, g, b, sigma) at world space points xyz.
        Please call encode first!
        :param xyz (SB, B, 3)
        SB is batch of objects
        B is batch of points (in rays)
        NS is number of input views
        :return (SB, B, 4) r g b sigma
        """
        with profiler.record_function("model_inference"):
            SB, B, _ = xyz.shape
            num_ray_per_obj = B//self.coarse_num if coarse else B//self.fine_num
            NS = self.num_views_per_obj
            xyz_world = xyz
            dir_world = viewdirs
            # Transform query points into the camera spaces of the input views
            xyz = repeat_interleave(xyz, NS)  # (SB*NS, B, 3)
            xyz_rot = torch.matmul(self.poses[:, None, :3, :3], xyz.unsqueeze(-1))[
                ..., 0
            ]
            xyz_cam = xyz_rot + self.poses[:, None, :3, 3]   # (SB*NS, num_ray*num_points, 3)

            # * Encode the xyz coordinates 
            if self.use_xyz:
                if self.normalize_z:
                    position = xyz_rot.reshape(-1, 3)  # (SB*B, 3)
                else:
                    position = xyz_world.reshape(-1, 3)  # (SB*B, 3)
            else:
                if self.normalize_z:
                    position = -xyz_rot[..., 2].reshape(-1, 1)  # (SB*B, 1)
                else:
                    position = -xyz[..., 2].reshape(-1, 1)  # (SB*B, 1)

            if self.use_code:
                # Positional encoding to position
                position = self.code(position)

            if self.use_viewdirs:
                viewdirs = viewdirs.reshape(SB, B, 3, 1)
                viewdirs = repeat_interleave(viewdirs, NS)  # (SB*NS, B, 3, 1)
                viewdirs_cam = torch.matmul(
                    self.poses[:, None, :3, :3], viewdirs
                )  # (SB*NS, B, 3, 1)
                direction = dir_world.reshape(-1, 3)  # (SB*B, 3)
                if self.use_code_viewdirs:
                    direction = self.code(direction)
                    
            
            z_feature = torch.cat(
                    (position, direction), dim=-1
                )  
            rela_dire = self.compute_angle(xyz, self.camera_location.to(xyz.device), viewdirs)
            if self.use_encoder:
                # Grab encoder's latent code.
                uv = -xyz_cam[:, :, :2] / xyz_cam[:, :, 2:]  # (SB, B, 2)
                uv *= repeat_interleave(
                    self.focal.unsqueeze(1), NS if self.focal.shape[0] > 1 else 1
                )
                uv += repeat_interleave(
                    self.c.unsqueeze(1), NS if self.c.shape[0] > 1 else 1
                )  # (SB*NS, B, 2)
                latent = self.encoder.index(
                    uv, None, self.image_shape
                )  # (SB*NS, latent_size, num_rays*num_points)

                if self.stop_encoder_grad:
                    latent = latent.detach()
                latent = latent.transpose(1, 2)  # shape: (SB*NS, num_rays*num_points, latent_size)
                latent = torch.cat((rela_dire, latent), dim=-1)
                if coarse:
                    latent = latent.reshape(SB, NS, -1, self.coarse_num, self.d_latent+7)
                else:                          # shape: (SB, num_views, num_rays, num_points, latent_size)
                    latent = latent.reshape(SB, NS, -1, self.fine_num, self.d_latent+7)
            if self.use_global_encoder:
                # Use to combine local feature (extracted by CNN) and global feature (extracted by ViT) to get a hybrid representation
                assert NotImplementedError('Not implemented yet')
            
            # Generate masking indexs
            if self.stage == 'pretrain' and self.mask_ratio > 0.0 and not coarse:
                # Define masking template
                mask_index = torch.zeros_like(latent)[..., 0] # shape: (SB, num_views, num_rays, num_points)
                mask_pt_on_a_ray = int(self.fine_num * self.mask_ratio)
                mask_pt_idx = torch.rand((SB, num_ray_per_obj, self.fine_num)).to(latent.device).argsort(dim = -1)[..., :mask_pt_on_a_ray] # shape: (SB, num_rays, mask_pt_num)
                mask_pt_idx = mask_pt_idx.unsqueeze(1).expand(-1, NS, -1, -1)   # shape: (SB, num_views, num_rays, mask_pt_num)
                target_mask = torch.rand_like(mask_pt_idx.float()).argsort(dim=1)  # shape: (SB, num_views, num_rays, mask_pt_num)
                view_idx = torch.randint(0, NS, (SB, 1, num_ray_per_obj, mask_pt_on_a_ray)).to(latent.device)    # shape: (SB, 1, num_rays, mask_pt_num)
                target_mask = (target_mask <= view_idx).float() # shape: (SB, num_views, num_rays, mask_pt_num)
                mask_index = mask_index.scatter(dim=-1, index=mask_pt_idx, src=target_mask) # shape: (SB, num_views, num_rays, num_points)
                mask_index = mask_index.unsqueeze(-1)    # shape: (SB, num_views, num_rays, num_points, 1)
            else:
                mask_index = None
            # Run main NeRF network
            if coarse or self.mlp_fine is None:
                mlp_output, inter_feat = self.mlp_coarse(
                    latent,
                    z_feature.reshape(SB, -1, self.coarse_num, self.d_in),
                    mask_index=None
                )   # mlp_output.shape: (SB, num_rays, num_points, 4)
                if self.stage == 'pretrain':
                    with torch.no_grad():
                        out_proj_target = self.target_projector(inter_feat.reshape(-1, inter_feat.shape[-1])).reshape(*inter_feat.shape[:-1], -1) # (SB, num_rays, num_points, projection_size)
                    useful_feat = out_proj_target
                else:
                    useful_feat = None
            else:
                mlp_output, inter_feat = self.mlp_fine(
                    latent,  # shape: (SB, num_views, num_rays, num_points, C1)
                    z_feature.reshape(SB, -1, self.fine_num, self.d_in), # shape: (SB, num_rays, num_points, C2)
                    mask_index = mask_index,    # shape: (SB, num_views, num_rays, num_points, 1)
                ) # inter_feat.shape: (SB, num_rays, num_points, hidden_size)
                if self.stage == 'pretrain':
                    out_proj = self.projector(inter_feat.reshape(-1, inter_feat.shape[-1]))
                    out_pred = self.predictor(out_proj).reshape(*inter_feat.shape[:-1], -1) # (SB, num_rays, num_points, projection_size)
                    useful_feat = out_pred
                else:
                    useful_feat = None
            rgb = mlp_output[..., :3]
            sigma = mlp_output[..., 3:4]

            output_list = [torch.sigmoid(rgb), torch.relu(sigma)]
            output = torch.cat(output_list, dim=-1)  # shape: (SB, num_rays, num_points, 4)
        
        return output, useful_feat

    def load_weights(self, args, opt_init=False, strict=True, device=None):
        """
        Helper for loading weights according to argparse arguments.
        Your can put a checkpoint at checkpoints/<exp>/nerformer_init to use as initialization.
        :param opt_init if true, loads from init checkpoint instead of usual even when resuming
        """
        # TODO: make backups
        if opt_init and not args.resume:
            return
        ckpt_name = (
            "nerformer_init" if opt_init or not args.resume else "nerformer_latest"
        )
        model_path = "%s/%s/%s" % (args.checkpoints_path, args.name, ckpt_name)

        if device is None:
            device = self.poses.device

        if os.path.exists(model_path):
            print("Load", model_path)
            self.load_state_dict(
                torch.load(model_path, map_location=device), strict=strict
            )
        elif not opt_init:
            warnings.warn(
                (
                    "WARNING: {} does not exist, not loaded!! Model will be re-initialized.\n"
                    + "If you are trying to load a pretrained model, STOP since it's not in the right place. "
                    + "If training, unless you are startin a new experiment, please remember to pass --resume."
                ).format(model_path)
            )
        return self

    def save_weights(self, args, opt_init=False):
        """
        Helper for saving weights according to argparse arguments
        :param opt_init if true, saves from init checkpoint instead of usual
        """
        from shutil import copyfile

        ckpt_name = "nerformer_init" if opt_init else "nerformer_latest"
        backup_name = "nerformer_init_backup" if opt_init else "nerformer_backup"

        ckpt_path = osp.join(args.checkpoints_path, args.name, ckpt_name)
        ckpt_backup_path = osp.join(args.checkpoints_path, args.name, backup_name)

        if osp.exists(ckpt_path):
            copyfile(ckpt_path, ckpt_backup_path)
        torch.save(self.state_dict(), ckpt_path)
        return self
    
    def updata_target(self, gamma=None):
        if gamma is None:
            gamma = self.moving_average_decay
        update_moving_average(gamma, self.target_projector, self.projector)
    
    def compute_angle(self, xyz, source_cam, dir_target):
        """
        :param xyz: (SB*NS, B, 3)
        :param source_cam: (SB*NS, 3, 1)
        :param dir_target: (SB*NS, B, 3, 1)
        :return: (SB*NS, B, 4); The first 3 channels are unit-length vector of the difference between
        query and target ray directions, the last channel is the inner product of the two directions.
        """
        
        ray2tar_pose = dir_target.clone()[...,0] 
        ray2tar_pose /= torch.norm(ray2tar_pose, dim=-1, keepdim=True) + 1e-6 # (SB*NS, B, 3)
        ray2train_pose = xyz - source_cam[:,None,:,0] # (SB*NS, B, 3)
        ray2train_pose /= torch.norm(ray2train_pose, dim=-1, keepdim=True) + 1e-6
        ray_diff = ray2tar_pose - ray2train_pose
        ray_diff_norm = torch.norm(ray_diff, dim=-1, keepdim=True)
        ray_diff_dot = torch.sum(ray2tar_pose * ray2train_pose, dim=-1, keepdim=True)
        ray_diff_direction = ray_diff / torch.clamp(ray_diff_norm, min=1e-6)
        ray_diff = torch.cat([ray_diff_direction, ray_diff_dot], dim=-1) # (SB*NS, B, 4)
        
        return ray_diff
