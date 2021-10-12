import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .activation import MyReLU
from .bone_utils import get_canonical_pose
from .stylegan import EqualLinear, EqualConv1d, NormalizedConv1d
from .model_utils import in_cube
from .utils_3d import SE3


class ResBlock(nn.Module):
    def __init__(self, hidden_dim, shrink_ratio=1, groups=1):
        super(ResBlock, self).__init__()
        self.c1 = NormalizedConv1d(hidden_dim, hidden_dim // shrink_ratio, 1, groups=groups)
        self.c2 = NormalizedConv1d(hidden_dim // shrink_ratio, hidden_dim, 1, groups=groups)
        self.act = nn.ReLU(inplace=True)

        self.hidden_dim = hidden_dim

    @property
    def memory_cost(self):
        m = 0
        for layer in self.children():
            if isinstance(layer, (NormalizedConv1d,)):
                m += layer.memory_cost
        m += self.hidden_dim
        return m

    @property
    def flops(self):
        fl = 0
        for layer in self.children():
            if isinstance(layer, (NormalizedConv1d,)):
                fl += layer.flops
            else:
                fl += self.hidden_dim
        fl += self.hidden_dim
        return fl

    def forward(self, x):
        h = x
        h = self.act(self.c1(h))
        h = self.act(self.c2(h))
        return x + h


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, groups=1, num_layers=8, final_activation=True):
        super(MLP, self).__init__()
        act = nn.ReLU(inplace=True)
        layers = [NormalizedConv1d(in_dim, hidden_dim, 1, groups=groups),
                  act]

        for _ in range((num_layers - 2) // 2):
            layers += [ResBlock(hidden_dim, groups=groups)]
        layers += [NormalizedConv1d(hidden_dim, out_dim, 1, groups=groups)]
        if final_activation:
            layers += [act]
        self.model = nn.Sequential(*layers)

        self.hidden_dim = hidden_dim

    @property
    def memory_cost(self):
        m = 0
        for layer in self.model.children():
            if isinstance(layer, (NormalizedConv1d, ResBlock)):
                m += layer.memory_cost
        return m

    @property
    def flops(self):
        fl = 0
        for layer in self.model.children():
            if isinstance(layer, (NormalizedConv1d, ResBlock)):
                fl += layer.flops
            else:
                fl += self.hidden_dim
        return fl

    def forward(self, x):
        return self.model(x)


class NeRF(nn.Module):
    def __init__(self, config, z_dim=256, num_bone=1, bone_length=True, parent=None, num_bone_param=None):
        super(NeRF, self).__init__()
        self.config = config
        hidden_size = config.hidden_size
        use_world_pose = not config.no_world_pose
        use_ray_direction = not config.no_ray_direction

        dim = 3  # xyz
        num_mlp_layers = 8
        self.out_dim = config.out_dim if "out_dim" in self.config else 3
        self.parent_id = parent
        self.use_bone_length = bone_length

        self.mask_input = self.config.concat and self.config.mask_input
        self.selector_activation = self.config.selector_activation
        selector_tmp = self.config.selector_adaptive_tmp.start
        self.register_buffer("selector_tmp", torch.tensor(selector_tmp).float())
        self.mask_before_PE = self.config.mask_before_PE

        if config.concat:
            self.save_mask = False  # flag to save mask
            self.groups = 1
        else:
            arf_temperature = config.arf_temperature
            self.arf_temperature = arf_temperature
            self.groups = num_bone

        self.density_activation = MyReLU.apply

        self.density_scale = config.density_scale

        # parameters for position encoding
        self.num_frequency_for_position = 10
        self.num_frequency_for_other = 4

        self.hidden_size = hidden_size
        self.num_bone = num_bone
        self.num_bone_param = num_bone_param if num_bone_param is not None else num_bone
        self.z_dim = z_dim
        if z_dim > 0:
            self.fc_z = EqualLinear(z_dim, hidden_size * self.groups)
            self.fc_c = EqualLinear(z_dim, hidden_size * self.groups)

        weighted_average_input = self.mask_input and (self.config.weighted_average or self.mask_before_PE)
        first_layer_num_bone = 1 if weighted_average_input else self.num_bone
        if bone_length:
            assert not self.config.weighted_average
            self.fc_bone_length = NormalizedConv1d(self.num_frequency_for_other * 2 * self.num_bone_param,
                                                   hidden_size * self.groups, 1)
        self.fc_p = NormalizedConv1d(dim * self.num_frequency_for_position * 2 * first_layer_num_bone,
                                     hidden_size * self.groups, 1, groups=self.groups)
        self.use_world_pose = use_world_pose
        self.use_ray_direction = use_ray_direction
        if use_world_pose:
            if self.config.se3:
                self.se3 = SE3()
            self.elements_in_translation_matrix = 6 if self.config.se3 else 12
            self.fc_Rt = NormalizedConv1d(
                self.elements_in_translation_matrix * self.num_frequency_for_other * 2 * first_layer_num_bone,
                hidden_size // 2 * self.groups, 1, groups=self.groups)

        if use_ray_direction:
            self.fc_d = NormalizedConv1d(dim * self.num_frequency_for_other * 2 * first_layer_num_bone,
                                         hidden_size // 2 * self.groups, 1, groups=self.groups)

        if self.mask_input:
            print("mask input")
            hidden_dim_for_mask = 10
            self.mask_linear_p = EqualConv1d(dim * self.num_frequency_for_position * 2 * self.num_bone,
                                             hidden_dim_for_mask * self.num_bone, 1, groups=self.num_bone)
            self.mask_linear_l = EqualConv1d(self.num_frequency_for_other * 2 * self.num_bone_param,
                                             hidden_dim_for_mask * self.num_bone, 1)
            self.mask_linear = EqualConv1d(hidden_dim_for_mask * self.num_bone,
                                           self.num_bone, 1, groups=self.num_bone)

        self.density_mlp = MLP(hidden_size * self.groups, hidden_size * self.groups,
                               hidden_size * self.groups, groups=self.groups, num_layers=num_mlp_layers)
        self.density_fc = NormalizedConv1d(hidden_size * self.groups, self.groups, 1, groups=self.groups)

        self.color_mlp = nn.Sequential(
            NormalizedConv1d(hidden_size * self.groups, hidden_size // 2 * self.groups, 1,
                             groups=self.groups),  # normal conv
        )

        self.color_fc = NormalizedConv1d(hidden_size // 2 * self.groups, self.out_dim * self.groups, 1,
                                         groups=self.groups)
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()

    @property
    def memory_cost(self):
        m = 0
        for layer in self.children():
            if isinstance(layer, (NormalizedConv1d, EqualLinear, EqualConv1d, MLP)):
                m += layer.memory_cost
        return m

    @property
    def flops(self):
        fl = 0
        for layer in self.children():
            if isinstance(layer, (NormalizedConv1d, EqualLinear, EqualConv1d, MLP)):
                fl += layer.flops

        if self.z_dim > 0:
            fl += self.hidden_size * 2
        if self.use_bone_length:
            fl += self.hidden_size
        fl += self.hidden_size * 2
        return fl

    def update_selector_tmp(self):
        gamma = self.config.selector_adaptive_tmp.gamma
        if gamma != 1:
            self.selector_tmp[()] = torch.clamp_min(self.selector_tmp * gamma,
                                                    self.config.selector_adaptive_tmp.min).float()

    def mask_x(self, x: torch.tensor, mask_prob: torch.tensor, num_bone=None) -> torch.tensor:
        batchsize, _, n = x.shape
        num_bone = num_bone or self.num_bone
        x = x.reshape(batchsize, num_bone, -1, n)
        x = x * mask_prob[:, :, None]
        if self.config.weighted_average or self.mask_before_PE:
            x = x.sum(dim=1)  # B x dim_position_encode x mask_prob.shape[2]
        else:  # weighted concat
            x = x.reshape(batchsize, -1, mask_prob.shape[2])
        return x

    def encode(self, value, num_frequency: int, num_bone=None):
        """
        positional encoding for group conv
        :param value: b x -1 x n
        :param num_frequency: L in NeRF paper
        :param num_bone: num_bone for positional encoding
        :return:
        """
        b, _, n = value.shape
        num_bone = num_bone or self.num_bone  # replace if None
        values = [2 ** i * value.reshape(b, num_bone, -1, n) * np.pi for i in range(num_frequency)]
        values = torch.cat(values, dim=2)
        gamma_p = torch.cat([torch.sin(values), torch.cos(values)], dim=2)
        gamma_p = gamma_p.reshape(b, -1, n)
        # mask outsize [-1, 1]
        mask = (value.reshape(b, num_bone, -1, n).abs() > 1).float().sum(dim=2, keepdim=True) >= 1
        mask = mask.float().repeat(1, 1, gamma_p.shape[1] // num_bone, 1)
        mask = mask.reshape(gamma_p.shape)
        return gamma_p * (1 - mask)  # B x (groups * ? * L * 2) x n

    def apply_mask(self, x, encoded_x, mask_prob, num_frequency=1, num_bone=None):
        if self.mask_before_PE:
            return self.encode(self.mask_x(x, mask_prob), num_frequency, 1)

        else:
            return self.mask_x(encoded_x, mask_prob, num_bone=num_bone)

    def backbone_(self, p, z=None, j=None, bone_length=None, ray_direction=None):
        batchsize, _, n = p.shape
        act = nn.LeakyReLU(0.2, inplace=True)
        if z is not None:
            z, c = torch.split(z, z.shape[1] // 2, dim=1)
            net_z = self.fc_z(z).unsqueeze(2)  # B x hidden_size x 1
            net_c = self.fc_c(c).unsqueeze(2)  # B x hidden_size x 1
        else:
            net_z = 0
            net_c = 0

        def clac_p_and_length_feature(p, bone_length):
            if bone_length is not None:
                encoded_length = self.encode(bone_length, self.num_frequency_for_other, num_bone=self.num_bone_param)
            encoded_p = self.encode(p, self.num_frequency_for_position)

            _mask_prob = None
            if self.mask_input:
                net = self.mask_linear_p(encoded_p)
                if bone_length is not None:
                    net = net + self.mask_linear_l(encoded_length)
                input_mask = self.mask_linear(F.relu(net, inplace=True))  # B x num_bone x n

                if self.selector_activation == "softmax":
                    _mask_prob = torch.softmax(input_mask / self.selector_tmp, dim=1)  # B x num_bone x n
                elif self.selector_activation == "sigmoid":
                    _mask_prob = torch.sigmoid(input_mask)  # B x num_bone x n
                else:
                    raise ValueError()

                if self.save_mask:  # save mask for segmentation rendering
                    self.mask_prob = _mask_prob.argmax(dim=1).data.cpu().numpy()  # B x n

                if self.config.use_scale_factor:
                    scale_factor = self.num_bone ** 0.5 / torch.norm(_mask_prob, dim=1, keepdim=True)
                    _mask_prob = _mask_prob * scale_factor

                if self.mask_before_PE:
                    p = p + get_canonical_pose(j)

                encoded_p = self.apply_mask(p, encoded_p, _mask_prob,
                                            self.num_frequency_for_position)  # mask position
                if bone_length is not None and self.config.mask_bone_length:
                    encoded_length = self.apply_mask(bone_length, encoded_length, _mask_prob,
                                                     self.num_frequency_for_other,
                                                     num_bone=self.num_bone_param)  # mask bone length

            net = self.fc_p(encoded_p)

            if bone_length is not None:
                net_bone_length = self.fc_bone_length(encoded_length)
            else:
                net_bone_length = 0

            return net + net_bone_length, _mask_prob

        net, mask_prob = clac_p_and_length_feature(p, bone_length)

        net = net_z + net

        if j is not None and self.use_world_pose:
            def calc_pose_feature(pose, mask_prob):
                if self.config.se3:
                    pose = self.se3.from_matrix(pose)  # B x num_bone x 6 x 1
                else:
                    pose = pose[:, :, :3]

                pose = pose.reshape(batchsize, self.num_bone * self.elements_in_translation_matrix, 1)
                encoded_pose = self.encode(pose, self.num_frequency_for_other)

                if self.mask_input:
                    encoded_pose = self.apply_mask(pose, encoded_pose, mask_prob,
                                                   self.num_frequency_for_other)

                net_pose = self.fc_Rt(encoded_pose)
                return net_pose

            net_pose = calc_pose_feature(j, mask_prob)
        else:
            net_pose = 0

        # ray direction
        if ray_direction is not None and self.use_ray_direction:
            assert n % ray_direction.shape[2] == 0

            def calc_ray_feature(ray_direction):
                ray_direction = ray_direction.unsqueeze(3).repeat(1, 1, 1, n // ray_direction.shape[2])
                ray_direction = ray_direction.reshape(batchsize, -1, n)
                encoded_d = self.encode(ray_direction, self.num_frequency_for_other)

                if self.mask_input:
                    encoded_d = self.apply_mask(ray_direction, encoded_d, mask_prob,
                                                self.num_frequency_for_other)

                net_d = self.fc_d(encoded_d)
                return net_d

            net_d = calc_ray_feature(ray_direction)

        else:
            net_d = 0

        net = self.density_mlp(net)
        # self.net_feat = net

        density = self.density_fc(net)  # B x 1 x n

        if self.density_scale != 1:
            density = density * self.density_scale

        # normalize feature
        net = F.normalize(net, dim=1) * self.hidden_size ** 0.5
        # add color latent
        net = act(net + net_c)

        net = self.color_mlp(net)

        # add pose and direction
        net = act(net + net_pose + net_d)
        net = self.color_fc(net)
        color = net.reshape(batchsize, self.groups, self.out_dim, n)
        density = self.density_activation(density)
        color = F.tanh(color)

        return density, color

    def backbone(self, p, z=None, j=None, bone_length=None, ray_direction=None):
        num_pixels = ray_direction.shape[2]  # number of sampled pixels
        chunk_size = self.config.max_chunk_size // p.shape[0]
        if num_pixels > chunk_size:
            num_points_on_a_ray = p.shape[2] // ray_direction.shape[2]
            density, color = [], []
            for i in range(0, num_pixels, chunk_size):
                p_chunk = p[:, :, i * num_points_on_a_ray:(i + chunk_size) * num_points_on_a_ray]
                ray_direction_chunk = ray_direction[:, :, i:i + chunk_size]
                bone_length.requires_grad = True
                density_i, color_i = torch.utils.checkpoint.checkpoint(self.backbone_, p_chunk, z, j,
                                                                       bone_length, ray_direction_chunk)
                density.append(density_i)
                color.append(color_i)

            density = torch.cat(density, dim=2)
            color = torch.cat(color, dim=3)
            return density, color

        else:
            return self.backbone_(p, z, j, bone_length, ray_direction)

    def calc_color_and_density(self, p, z=None, pose_world=None, bone_length=None, ray_direction=None):
        """
        forward func of ImplicitField
        :param pose_world:
        :param p: b x groups * 3 x n (n = num_of_ray * points_on_ray)
        :param z: b x dim
        :param pose_world: b x groups x 4 x 4
        :param bone_length: b x groups x 1
        :param ray_direction: b x groups * 3 x m (m = number of ray)
        :return: b x groups x 4 x n
        """
        density, color = self.backbone(p, z, pose_world, bone_length=bone_length, ray_direction=ray_direction)
        if not self.config.concat:
            # density is zero if p is outside the cube
            density *= in_cube(p)
        return density, color  # B x groups x 1 x n, B x groups x 3 x n

    @staticmethod
    def coord_transform(p: torch.tensor, rotation: torch.tensor, translation: torch.tensor) -> torch.tensor:
        # 'world coordinate' -> 'bone coordinate'
        return torch.matmul(rotation.permute(0, 2, 1), p - translation)

    def sum_density(self, density: torch.tensor, semantic_map: bool = False) -> (torch.tensor,) * 2:
        """

        :param density: B x num_bone x 1 x n x N
        :param semantic_map:
        :return:
        """
        temperature = 100 if semantic_map else self.arf_temperature
        alpha = torch.softmax(density * temperature, dim=1)  # B x num_bone x 1 x n x Nc-1

        if self.config.detach_alpha:
            alpha = alpha.detach()

        # sum density across bone
        if self.config.sum_density:
            density = density.sum(dim=1, keepdim=True)
        else:
            density = (density * alpha).sum(dim=1, keepdim=True)
        return density, alpha

    def coarse_to_fine_sample(self, image_coord: torch.tensor, pose_to_camera: torch.tensor,
                              inv_intrinsics: torch.tensor, z: torch.tensor = None, world_pose: torch.tensor = None,
                              bone_length: torch.tensor = None, near_plane: float = 0.3, far_plane: float = 5,
                              Nc: int = 64, Nf: int = 128, render_scale: float = 1) -> (torch.tensor,) * 3:
        n_samples_to_decide_depth_range = 16
        batchsize, _, _, n = image_coord.shape
        num_bone = 1 if self.config.concat_pose else self.num_bone  # PoseConditionalNeRF or other
        with torch.no_grad():
            if self.config.concat_pose:  # recompute pose of camera
                pose_to_camera = torch.matmul(pose_to_camera, torch.inverse(world_pose))[:, :1]
            # rotation & translation
            R = pose_to_camera[:, :, :3, :3].reshape(batchsize * num_bone, 3, 3)  # B*num_bone x 3 x 3
            t = pose_to_camera[:, :, :3, 3].reshape(batchsize * num_bone, 3, 1)  # B*num_bone x 3 x 1

            if inv_intrinsics.ndim == 2:
                image_coord = image_coord.reshape(batchsize * num_bone, 3, n)
                # img coord -> camera coord
                sampled_camera_coord = torch.matmul(inv_intrinsics, image_coord)
            else:
                # reshape for multiplying inv_intrinsics
                image_coord = image_coord.reshape(batchsize, num_bone, 3, n)
                image_coord = image_coord.permute(0, 2, 1, 3)  # B x 3 x num_bone x n
                image_coord = image_coord.reshape(batchsize, 3, num_bone * n)

                # img coord -> camera coord
                sampled_camera_coord = torch.matmul(inv_intrinsics, image_coord)
                sampled_camera_coord = sampled_camera_coord.reshape(batchsize, 3, num_bone, n)
                sampled_camera_coord = sampled_camera_coord.permute(0, 2, 1, 3)
                sampled_camera_coord = sampled_camera_coord.reshape(batchsize * num_bone, 3, n)

            # camera coord -> bone coord
            sampled_bone_coord = self.coord_transform(sampled_camera_coord, R, t)  # B*num_bone x 3 x n

            # camera origin
            camera_origin = self.coord_transform(torch.zeros_like(sampled_camera_coord), R, t)  # B*num_bone x 3 x n

            # ray direction
            ray_direction = sampled_bone_coord - camera_origin  # B*num_bone x 3 x n

            # unit ray direction
            ray_direction = F.normalize(ray_direction, dim=1)  # B*num_bone x 3 x n

            # sample points to decide depth range
            sampled_depth = torch.linspace(near_plane, far_plane, n_samples_to_decide_depth_range, device="cuda")
            sampled_points_on_rays = camera_origin[:, :, :, None] + ray_direction[:, :, :, None] * sampled_depth

            # inside the cube [-1, 1]^3?
            inside = in_cube(sampled_points_on_rays)  # B*num_bone x 1 x n x n_samples_to_decide_depth_range

            # minimum-maximum depth
            depth_min = torch.where(inside, sampled_depth * inside,
                                    torch.full_like(inside.float(), 1e3)).min(dim=3)[0]
            depth_max = torch.where(inside, sampled_depth * inside,
                                    torch.full_like(inside.float(), -1e3)).max(dim=3)[0]
            # # replace values if no intersection
            depth_min = torch.where(inside.sum(dim=3) > 0, depth_min, torch.full_like(depth_min, near_plane))
            depth_max = torch.where(inside.sum(dim=3) > 0, depth_max, torch.full_like(depth_max, far_plane))

            # adopt the smallest/largest values among bones
            depth_min = depth_min.reshape(batchsize, num_bone, 1, n).min(dim=1, keepdim=True)[0]  # B x 1 x 1 x n
            depth_max = depth_max.reshape(batchsize, num_bone, 1, n).max(dim=1, keepdim=True)[0]

            start = (camera_origin.reshape(batchsize, num_bone, 3, n) +
                     depth_min * ray_direction.reshape(batchsize, num_bone, 3, n))  # B x num_bone x 3 x n
            end = (camera_origin.reshape(batchsize, num_bone, 3, n) +
                   depth_max * ray_direction.reshape(batchsize, num_bone, 3, n))  # B x num_bone x 3 x n

            # coarse ray sampling
            bins = (torch.arange(Nc, dtype=torch.float, device="cuda").reshape(1, 1, 1, 1, Nc) / Nc +
                    torch.cuda.FloatTensor(batchsize, 1, 1, n, Nc).uniform_() / Nc)
            coarse_points = start.unsqueeze(4) * (1 - bins) + end.unsqueeze(4) * bins  # B x num_bone x 3 x n x Nc
            coarse_depth = (depth_min.unsqueeze(4) * (1 - bins) +
                            depth_max.unsqueeze(4) * bins)  # B x 1 x 1 x n x Nc

            ray_direction = ray_direction.reshape(batchsize, num_bone * 3, n)

            # coarse density
            coarse_density = self.calc_color_and_density(coarse_points.reshape(batchsize, num_bone * 3, n * Nc),
                                                         z, world_pose,
                                                         bone_length,
                                                         ray_direction)[0]  # B x groups x n*Nc

            if self.groups > 1:
                # alpha blending
                coarse_density, _ = self.sum_density(coarse_density)

            # calculate weight for fine sampling
            coarse_density = coarse_density.reshape(batchsize, 1, 1, n, Nc)[:, :, :, :, :-1]
            # # delta = distance between adjacent samples
            delta = coarse_depth[:, :, :, :, 1:] - coarse_depth[:, :, :, :, :-1]  # B x 1 x 1 x n x Nc-1

            density_delta = coarse_density * delta * render_scale
            T_i = torch.exp(-(torch.cumsum(density_delta, dim=4) - density_delta))
            weights = T_i * (1 - torch.exp(-density_delta))  # B x 1 x 1 x n x Nc-1
            weights = weights.reshape(batchsize * n, Nc - 1)
            # fine ray sampling
            bins = (torch.multinomial(torch.clamp_min(weights, 1e-8),
                                      Nf, replacement=True).reshape(batchsize, 1, 1, n, Nf).float() / Nc +
                    torch.cuda.FloatTensor(batchsize, 1, 1, n, Nf).uniform_() / Nc)
            fine_points = start.unsqueeze(4) * (1 - bins) + end.unsqueeze(4) * bins  # B x num_bone x 3 x n x Nf
            fine_depth = (depth_min.unsqueeze(4) * (1 - bins) +
                          depth_max.unsqueeze(4) * bins)  # B x 1 x 1 x n x Nc

            # sort points
            fine_points = torch.cat([coarse_points, fine_points], dim=4)
            fine_depth = torch.cat([coarse_depth, fine_depth], dim=4)
            arg = torch.argsort(fine_depth, dim=4)

            fine_points = torch.gather(fine_points, dim=4,
                                       index=arg.repeat(1, num_bone, 3, 1, 1))  # B x num_bone x 3 x n x Nc+Nf
            fine_depth = torch.gather(fine_depth, dim=4, index=arg)  # B x 1 x 1 x n x Nc+Nf

            fine_points = fine_points.reshape(batchsize, num_bone * 3, n * (Nc + Nf))

        if pose_to_camera.requires_grad:
            R = pose_to_camera[:, :, :3, :3]
            t = pose_to_camera[:, :, :3, 3:]

            with torch.no_grad():
                fine_points = fine_points.reshape(batchsize, num_bone, 3, n * (Nc + Nf))
                fine_points = torch.matmul(R, fine_points) + t
            fine_points = torch.matmul(R.permute(0, 1, 3, 2), fine_points - t).reshape(batchsize, num_bone * 3,
                                                                                       n * (Nc + Nf))
        return (
            fine_depth,  # B x 1 x 1 x n x Nc+Nf
            fine_points,  # B x num_bone*3 x n*Nc+Nf
            ray_direction  # B x num_bone*3 x n
        )

    def render(self, image_coord: torch.tensor, pose_to_camera: torch.tensor, inv_intrinsics: torch.tensor,
               z: torch.tensor = None, world_pose: torch.tensor = None, bone_length: torch.tensor = None,
               thres: float = 0.9, render_scale: float = 1, Nc: int = 64, Nf: int = 128,
               semantic_map: bool = False) -> (torch.tensor,) * 3:
        near_plane = 0.3
        # n <- number of sampled pixels
        # image_coord: B x groups x 3 x n
        # camera_extrinsics: B x 4 x 4
        # camera_intrinsics: 3 x 3

        batchsize, num_bone, _, n = image_coord.shape

        fine_depth, fine_points, ray_direction = self.coarse_to_fine_sample(image_coord, pose_to_camera,
                                                                            inv_intrinsics,
                                                                            z=z, world_pose=world_pose,
                                                                            bone_length=bone_length,
                                                                            near_plane=near_plane, Nc=Nc, Nf=Nf,
                                                                            render_scale=render_scale)
        # fine density & color # B x groups x 1 x n*(Nc+Nf), B x groups x 3 x n*(Nc+Nf)
        if semantic_map and self.config.mask_input:
            self.save_mask = True

        fine_density, fine_color = self.calc_color_and_density(fine_points, z, world_pose,
                                                               bone_length, ray_direction)
        if semantic_map and self.config.mask_input:
            self.save_mask = False

        # semantic map
        if semantic_map and not self.config.concat:
            bone_idx = torch.arange(num_bone).cuda()
            fine_color = torch.stack([bone_idx // 9, (bone_idx // 3) % 3, bone_idx % 3], dim=1) - 1  # num_bone x 3
            fine_color[::2] = fine_color.flip(dims=(0,))[1 - num_bone % 2::2]
            fine_color = fine_color[None, :, :, None, None]
        elif semantic_map and self.config.mask_input:
            bone_idx = torch.arange(num_bone).cuda()
            seg_color = torch.stack([bone_idx // 9, (bone_idx // 3) % 3, bone_idx % 3], dim=1) - 1  # num_bone x 3
            seg_color[::2] = seg_color.flip(dims=(0,))[1 - num_bone % 2::2]  # num_bone x 3
            fine_color = seg_color[self.mask_prob.reshape(-1)]
            fine_color = fine_color.reshape(batchsize, 1, -1, 3).permute(0, 1, 3, 2)
            fine_color = fine_color.reshape(batchsize, 1, 3, n, Nc + Nf)[:, :, :, :, :-1]
        else:
            fine_color = fine_color.reshape(batchsize, self.groups, self.out_dim, n, Nc + Nf)[:, :, :, :, :-1]

        fine_density = fine_density.reshape(batchsize, self.groups, 1, n, Nc + Nf)[:, :, :, :, :-1]

        if self.groups == 1:
            sum_fine_density = fine_density
        else:
            # alpha blending
            sum_fine_density, alpha = self.sum_density(fine_density, semantic_map=semantic_map)

            fine_color = (fine_color * alpha).sum(dim=1, keepdims=True)

        if thres > 0:
            # density = inf if density exceeds thres
            sum_fine_density = (sum_fine_density > thres) * 100000

        delta = fine_depth[:, :, :, :, 1:] - fine_depth[:, :, :, :, :-1]  # B x 1 x 1 x n x Nc+Nf-1
        sum_density_delta = sum_fine_density * delta * render_scale  # B x 1 x 1 x n x Nc+Nf-1

        T_i = torch.exp(-(torch.cumsum(sum_density_delta, dim=4) - sum_density_delta))
        weights = T_i * (1 - torch.exp(-sum_density_delta))  # B x 1 x 1 x n x Nc+Nf-1

        fine_depth = fine_depth.reshape(batchsize, 1, 1, n, Nc + Nf)[:, :, :, :, :-1]

        rendered_color = torch.sum(weights * fine_color, dim=4).squeeze(1)  # B x 3 x n
        rendered_mask = torch.sum(weights, dim=4).reshape(batchsize, n)  # B x n
        rendered_disparity = torch.sum(weights * 1 / fine_depth, dim=4).reshape(batchsize, n)  # B x n

        return rendered_color, rendered_mask, rendered_disparity

    def forward(self, batchsize, num_sample, sampled_img_coord, pose_to_camera, inv_intrinsics, z=None,
                world_pose=None, bone_length=None, thres=0.9, render_scale=1, Nc=64, Nf=128):
        """
        rendering function for sampled rays
        :param batchsize:
        :param num_sample:
        :param sampled_img_coord: sampled image coordinate
        :param pose_to_camera:
        :param inv_intrinsics:
        :param z:
        :param world_pose:
        :param bone_length:
        :param thres:
        :param render_scale:
        :param Nc:
        :param Nf:
        :return: color and mask value for sampled rays
        """

        # repeat coords along bone axis
        if not self.config.concat_pose:
            sampled_img_coord = sampled_img_coord.repeat(1, self.num_bone, 1, 1)

        merged_color, merged_mask, _ = self.render(sampled_img_coord,
                                                   pose_to_camera,
                                                   inv_intrinsics,
                                                   z=z,
                                                   world_pose=world_pose,
                                                   bone_length=bone_length,
                                                   thres=thres,
                                                   Nc=Nc,
                                                   Nf=Nf,
                                                   render_scale=render_scale)

        return merged_color, merged_mask

    def render_entire_img(self, pose_to_camera, inv_intrinsics, z=None, world_pose=None, bone_length=None,
                          thres=0.9, render_scale=1, batchsize=1000, render_size=128, Nc=64, Nf=128,
                          semantic_map=False, use_normalized_intrinsics=False):
        assert z is None or z.shape[0] == 1
        assert bone_length is None or bone_length.shape[0] == 1
        assert world_pose is None or world_pose.shape[0] == 1
        batchsize = self.config.render_bs or batchsize
        if use_normalized_intrinsics:
            img_coord = torch.stack([(torch.arange(render_size * render_size) % render_size + 0.5) / render_size,
                                     (torch.arange(render_size * render_size) // render_size + 0.5) / render_size,
                                     torch.ones(render_size * render_size).long()], dim=0).float()
        else:
            img_coord = torch.stack([torch.arange(render_size * render_size) % render_size + 0.5,
                                     torch.arange(render_size * render_size) // render_size + 0.5,
                                     torch.ones(render_size * render_size).long()], dim=0).float()

        img_coord = img_coord[None, None].cuda()

        if not self.config.concat_pose:
            img_coord = img_coord.repeat(1, self.num_bone, 1, 1)

        rendered_color = []
        rendered_mask = []
        rendered_disparity = []

        with torch.no_grad():
            for i in range(0, render_size ** 2, batchsize):
                (rendered_color_i, rendered_mask_i,
                 rendered_disparity_i) = self.render(img_coord[:, :, :, i:i + batchsize],
                                                     pose_to_camera[:1],
                                                     inv_intrinsics,
                                                     z=z,
                                                     bone_length=bone_length,
                                                     world_pose=world_pose,
                                                     thres=thres,
                                                     render_scale=render_scale, Nc=Nc, Nf=Nf,
                                                     semantic_map=semantic_map)
                rendered_color.append(rendered_color_i)
                rendered_mask.append(rendered_mask_i)
                rendered_disparity.append(rendered_disparity_i)

            rendered_color = torch.cat(rendered_color, dim=2)
            rendered_mask = torch.cat(rendered_mask, dim=1)
            rendered_disparity = torch.cat(rendered_disparity, dim=1)

        return (rendered_color.reshape(3, render_size, render_size),  # 3 x size x size
                rendered_mask.reshape(render_size, render_size),  # size x size
                rendered_disparity.reshape(render_size, render_size))  # size x size


class PoseConditionalNeRF(NeRF):
    def __init__(self, config, z_dim=256, num_bone=1, bone_length=True, parent=None):
        super(NeRF, self).__init__()
        self.config = config
        hidden_size = config.hidden_size
        use_world_pose = not config.no_world_pose
        use_ray_direction = not config.no_ray_direction
        self.use_bone_length = bone_length
        dim = 3  # xyz

        self.groups = 1

        self.density_activation = MyReLU.apply

        self.density_scale = config.density_scale

        # parameters for position encoding
        self.num_frequency_for_position = 10
        self.num_frequency_for_other = 4

        self.hidden_size = hidden_size
        self.num_bone = num_bone
        self.z_dim = z_dim
        if z_dim > 0:
            self.fc_z = EqualLinear(z_dim, hidden_size * self.groups)
            self.fc_c = EqualLinear(z_dim, hidden_size * self.groups)

        if bone_length:
            self.fc_bone_length = NormalizedConv1d(self.num_frequency_for_other * 2 * self.num_bone,
                                                   hidden_size, 1)
        self.fc_p = NormalizedConv1d(dim * self.num_frequency_for_position * 2,
                                     hidden_size, 1)
        self.fc_d = NormalizedConv1d(dim * self.num_frequency_for_other * 2,
                                     hidden_size // 2, 1)
        self.use_world_pose = use_world_pose
        self.use_ray_direction = use_ray_direction
        if use_world_pose:
            if self.config.se3:
                self.se3 = SE3()
            self.elements_in_translation_matrix = 6 if self.config.se3 else 12
            self.fc_Rt = NormalizedConv1d(
                self.elements_in_translation_matrix * self.num_frequency_for_other * 2 * self.num_bone,
                hidden_size, 1)

        self.density_mlp = MLP(hidden_size, hidden_size, hidden_size, num_layers=8)
        self.density_fc = NormalizedConv1d(hidden_size, 1, 1)

        self.color_mlp = nn.Sequential(
            NormalizedConv1d(hidden_size, hidden_size // 2, 1),  # normal conv
        )

        self.color_fc = NormalizedConv1d(hidden_size // 2, 3, 1)
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()

    @property
    def memory_cost(self):
        m = 0
        for layer in self.children():
            if isinstance(layer, (NormalizedConv1d, EqualLinear, EqualConv1d, MLP)):
                m += layer.memory_cost
        return m

    @property
    def flops(self):
        fl = 0
        for layer in self.children():
            if isinstance(layer, (NormalizedConv1d, EqualLinear, EqualConv1d, MLP)):
                fl += layer.flops
        if self.z_dim > 0:
            fl += self.hidden_size * 2
        if self.use_bone_length:
            fl += self.hidden_size
        fl += self.hidden_size * 2
        return fl

    def backbone_(self, p, z=None, j=None, bone_length=None, ray_direction=None):
        batchsize, _, n = p.shape
        act = nn.LeakyReLU(0.2, inplace=True)
        if z is not None:
            z, c = torch.split(z, z.shape[1] // 2, dim=1)
            net_z = self.fc_z(z).unsqueeze(2)  # B x hidden_size x 1
            net_c = self.fc_c(c).unsqueeze(2)  # B x hidden_size x 1
        else:
            net_z = 0
            net_c = 0

        encoded_p = self.encode(p, self.num_frequency_for_position, num_bone=1)

        net = self.fc_p(encoded_p)

        if bone_length is not None:
            encoded_length = self.encode(bone_length, self.num_frequency_for_other, num_bone=self.num_bone_param)
            net_bone_length = self.fc_bone_length(encoded_length)
        else:
            net_bone_length = 0

        if j is not None and self.use_world_pose:
            def calc_pose_feature(pose):
                if self.config.se3:
                    pose = self.se3.from_matrix(pose)  # B x num_bone x 6 x 1
                else:
                    pose = pose[:, :, :3]
                encoded_pose = self.encode(
                    pose.reshape(batchsize, self.num_bone * self.elements_in_translation_matrix, 1),
                    self.num_frequency_for_other)

                _net_pose = self.fc_Rt(encoded_pose)
                return _net_pose

            net_pose = calc_pose_feature(j)
        else:
            net_pose = 0

        net = net_z + net + net_bone_length + net_pose

        # ray direction
        if ray_direction is not None and self.use_ray_direction:
            assert n % ray_direction.shape[2] == 0

            def calc_ray_feature(_ray_direction):
                _ray_direction = _ray_direction.unsqueeze(3).repeat(1, 1, 1, n // _ray_direction.shape[2])
                _ray_direction = _ray_direction.reshape(batchsize, -1, n)
                encoded_d = self.encode(_ray_direction, self.num_frequency_for_other, num_bone=1)

                net_d = self.fc_d(encoded_d)
                return net_d

            net_d = calc_ray_feature(ray_direction)

        else:
            net_d = 0

        net = self.density_mlp(net)

        density = self.density_fc(net)  # B x 1 x n

        if self.density_scale != 1:
            density = density * self.density_scale

        # normalize feature
        net = F.normalize(net, dim=1) * self.hidden_size ** 0.5
        # add color latent
        net = act(net + net_c)

        net = self.color_mlp(net)

        # add pose and direction
        net = act(net + net_d)
        net = self.color_fc(net)
        color = net.reshape(batchsize, self.groups, 3, n)
        density = self.density_activation(density)
        color = F.tanh(color)

        return density, color
