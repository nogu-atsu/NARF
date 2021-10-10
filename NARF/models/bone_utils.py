import torch


def get_canonical_pose(self, pose):  # B x num_bone x 4 x 4
    R = pose[:, self.parent_id[1:], :3, :3]
    t_pa = pose[:, self.parent_id[1:], :3, 3]
    t_ch = pose[:, 1:, :3, 3]
    t_diff = torch.matmul(R.permute(0, 1, 3, 2), (t_ch - t_pa)[:, :, :, None])
    canonical_t = [pose.new_zeros(pose.shape[0], 3, 1)]
    for i in range(1, pose.shape[1]):
        canonical_t.append(canonical_t[self.parent_id[i]] + t_diff[:, i - 1])
    canonical_t = torch.cat(canonical_t, dim=1)
    return canonical_t  # B x num_bone*3 x 1
