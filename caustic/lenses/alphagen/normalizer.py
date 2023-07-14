import ast

import h5py
import numpy as np
import torch
from scipy.interpolate import splrep

__all__ = ("Normalizer",)


class Normalizer:
    def __init__(self, normalize, mdef=None, segment="FG", stats_path=None):
        """
        mdef == "PJAFFE" -> hset_params = ['center_x', 'center_y', 'Ra', 'Rs', 'sigma0', 'z']
        mdef == "NFW" -> hset_params = ["alpha_Rs", "Rs", "center_x", "center_y", "z"]
        """
        self.normalize = normalize
        self.mdef = mdef
        self.segment = segment
        if self.normalize == "from_stats":
            try:
                stats_file = h5py.File(stats_path, mode="r")
                self.cone_fov = ast.literal_eval(
                    stats_file["base"].attrs["dataset_descriptor"]
                )["cone_fov"]

                self.z = torch.tensor(stats_file["base"]["z"][:], dtype=torch.float)  # type: ignore
                self.z_norm = torch.tensor(stats_file["base"]["z_norm"][:], dtype=torch.float).reshape(1, -1, 1, 1)  # type: ignore

                self.z_norm_fn = splrep(
                    np.insert(self.z.squeeze().detach().cpu().numpy(), 0, 0.0),
                    np.insert(self.z_norm.squeeze().detach().cpu().numpy(), 0, 0.0),
                )

                self.hset_bounds = torch.tensor(stats_file["base"]["hset_bounds"][:], dtype=torch.float)  # type: ignore
                if self.mdef == "PJAFFE":
                    self.log_slice = slice(2, -1)
                    self.hset_bounds[:, :2] = torch.tile(
                        torch.tensor(
                            [-self.cone_fov / 2, self.cone_fov / 2], dtype=torch.float
                        ),
                        (2, 1),
                    ).T  # bounds for halo positions are known
                    self.hset_bounds[:, self.log_slice] = torch.log10(
                        self.hset_bounds[:, 2:-1]
                    )  # take log of non-position dimensions except z
                elif self.mdef == "NFW":
                    self.log_slice = slice(2)
                    self.hset_bounds[:, 2:4] = torch.tile(
                        torch.tensor(
                            [-self.cone_fov / 2, self.cone_fov / 2], dtype=torch.float
                        ),
                        (2, 1),
                    ).T  # bounds for halo positions are known
                    self.hset_bounds[:, self.log_slice] = torch.log10(
                        self.hset_bounds[:, :2]
                    )  # take log of non-position dimensions except z

                self.reverse_alpha_norm = self.reverse_alpha_from_stats

                if self.segment == "FG":
                    self.forward_norm = self.forward_from_stats_FG
                    self.reverse_norm = self.reverse_from_stats_FG
                    self.forward_x_norm = self.forward_x_FG
                    self.reverse_x_norm = self.reverse_x_FG

                elif self.segment == "BG" or self.segment == "FULL":
                    self.a_LP_bound = stats_file["base"]["a_LP_bound"][()]  # type: ignore
                    self.beta_LP_bound = stats_file["base"]["beta_LP_bound"][()]  # type: ignore

                    self.forward_norm = self.forward_from_stats_BG
                    self.reverse_norm = self.reverse_from_stats_BG
                    self.forward_x_norm = self.forward_x_BG
                    self.reverse_x_norm = self.reverse_x_BG
            except FileNotFoundError:
                raise FileNotFoundError(f"stats file {stats_path} does not exist")

        elif self.normalize == None:
            self.forward_norm = self.forward_null
            self.reverse_norm = self.reverse_null
            self.reverse_alpha_norm = self.reverse_alpha_null
            self.forward_x_norm = self.forward_x_null
            self.reverse_x_norm = self.reverse_x_null

    def forward(self, *args, **kwargs):
        return self.forward_norm(*args, **kwargs)

    def reverse(self, *args, **kwargs):
        return self.reverse_norm(*args, **kwargs)

    def reverse_alpha(self, a, plane_ids=None):
        return self.reverse_alpha_norm(a, plane_ids)

    def forward_x(self, x):
        return self.forward_x_norm(x)

    def reverse_x(self, x):
        return self.reverse_x_norm(x)

    def reverse_alpha_from_stats(self, a, plane_ids):
        return a * self.z_norm[:, plane_ids].to(a.device)

    def forward_x_FG(self, x):
        x_scaled = self._min_max_scale(
            x, bounds=(-self.cone_fov / 2, self.cone_fov / 2)
        )
        return x_scaled

    def reverse_x_FG(self, x_scaled):
        x = self._min_max_unscale(
            x_scaled, bounds=(-self.cone_fov / 2, self.cone_fov / 2)
        )
        return x

    def forward_x_BG(self, a_LP, beta_LP):
        a_LP_scaled = self._min_max_scale(a_LP, bounds=(-self.a_LP_bound, self.a_LP_bound))  # type: ignore
        beta_LP_scaled = self._min_max_scale(beta_LP, bounds=(-self.beta_LP_bound, self.beta_LP_bound))  # type: ignore

        return a_LP_scaled, beta_LP_scaled

    def reverse_x_BG(self, a_LP_scaled, beta_LP_scaled):
        a_LP = self._min_max_unscale(a_LP_scaled, bounds=(-self.a_LP_bound, self.a_LP_bound))  # type: ignore
        beta_LP = self._min_max_unscale(beta_LP_scaled, bounds=(-self.beta_LP_bound, self.beta_LP_bound))  # type: ignore

        return a_LP, beta_LP

    def forward_from_stats_FG(self, hset, a=None):
        hset_scaled = torch.clone(hset)
        hset_scaled[..., self.log_slice] = torch.log10(hset_scaled[..., self.log_slice])
        hset_scaled[..., :-1] = self._min_max_scale(
            hset_scaled[..., :-1], bounds=self.hset_bounds[:, :-1]
        )
        if a is not None:
            a_scaled = self._safe_divide_z_norm(a, self.z_norm)
        else:
            a_scaled = None
        return hset_scaled, a_scaled

    def reverse_from_stats_FG(self, hset_scaled, a_scaled=None):
        hset = torch.clone(hset_scaled)
        hset[..., :-1] = self._min_max_unscale(
            hset[..., :-1], bounds=self.hset_bounds[:, :-1]
        )
        hset[..., self.log_slice] = 10 ** hset[..., self.log_slice]
        if a_scaled is not None:
            a = self.z_norm * a_scaled
        else:
            a = None
        return hset, a

    def forward_from_stats_BG(self, hset, a_LP, beta_LP, a=None):
        hset_scaled = torch.clone(hset)
        hset_scaled[..., self.log_slice] = torch.log10(hset_scaled[..., self.log_slice])
        hset_scaled[..., :-1] = self._min_max_scale(
            hset_scaled[..., :-1], bounds=self.hset_bounds[:, :-1]
        )

        a_LP_scaled = self._min_max_scale(a_LP, bounds=(-self.a_LP_bound, self.a_LP_bound))  # type: ignore
        beta_LP_scaled = self._min_max_scale(beta_LP, bounds=(-self.beta_LP_bound, self.beta_LP_bound))  # type: ignore

        if a is not None:
            a_scaled = self._safe_divide_z_norm(a, self.z_norm)
        else:
            a_scaled = None
        return hset_scaled, a_LP_scaled, beta_LP_scaled, a_scaled

    def reverse_from_stats_BG(
        self, hset_scaled, a_LP_scaled, beta_LP_scaled, a_scaled=None
    ):
        hset = torch.clone(hset_scaled)
        hset[..., :-1] = self._min_max_unscale(
            hset[..., :-1], bounds=self.hset_bounds[:, :-1]
        )
        hset[..., self.log_slice] = 10 ** hset[..., self.log_slice]

        a_LP = self._min_max_unscale(a_LP_scaled, bounds=(-self.a_LP_bound, self.a_LP_bound))  # type: ignore
        beta_LP = self._min_max_unscale(beta_LP_scaled, bounds=(-self.beta_LP_bound, self.beta_LP_bound))  # type: ignore

        if a_scaled is not None:
            a = self.z_norm * a_scaled
        else:
            a = None
        return hset, a_LP, beta_LP, a

    def forward_null(self, *args):
        return args

    def reverse_null(self, *args):
        return args

    def reverse_alpha_null(self, a, *args):
        return a

    def forward_x_null(self, x):
        return x

    def reverse_x_null(self, x):
        return x

    @staticmethod
    def _min_max_scale(arr, bounds, vrange=(-1, 1)):
        return vrange[0] + (arr - bounds[0]) * (vrange[1] - vrange[0]) / (
            bounds[1] - bounds[0]
        )

    @staticmethod
    def _min_max_unscale(arr, bounds, vrange=(-1, 1)):
        return (arr - vrange[0]) * (bounds[1] - bounds[0]) / (
            vrange[1] - vrange[0]
        ) + bounds[0]

    @staticmethod
    def _safe_divide_z_norm(num, denom):
        out = torch.zeros_like(num, dtype=torch.float)
        out[:, torch.squeeze(denom) != 0, ...] = (
            num[:, torch.squeeze(denom) != 0, ...]
            / denom[:, torch.squeeze(denom) != 0, ...]
        )
        return out
