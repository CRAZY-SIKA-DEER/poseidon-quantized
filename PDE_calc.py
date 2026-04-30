import numpy as np
import torch
import h5py
from transformers import AutoConfig
import copy
import torch.nn.functional as F


# ==================== Physical Loss Classes (Fixed) ====================

class PhysicalLossBase:
    """Base class for physical loss calculations."""
    def __init__(self, constants, Re=None, dt=None, dx=1.0/128, dy=1.0/128):
        self.constants = constants
        self.Re = Re
        self.dt = dt
        self.dx = dx
        self.dy = dy

    def denormalize(self, tensor):
        if isinstance(tensor, np.ndarray):
            mean = np.array(self.constants["mean"], dtype=tensor.dtype).flatten()
            std = np.array(self.constants["std"], dtype=tensor.dtype).flatten()
            if tensor.ndim == 3:
                mean = mean.reshape(-1, 1, 1)
                std = std.reshape(-1, 1, 1)
            elif tensor.ndim == 4:
                mean = mean.reshape(1, -1, 1, 1)
                std = std.reshape(1, -1, 1, 1)
            return tensor * std + mean
        elif isinstance(tensor, torch.Tensor):
            device = tensor.device
            mean = torch.as_tensor(self.constants["mean"], dtype=tensor.dtype, device=device).flatten()
            std = torch.as_tensor(self.constants["std"], dtype=tensor.dtype, device=device).flatten()
            if tensor.ndim == 3:
                mean = mean.view(-1, 1, 1)
                std = std.view(-1, 1, 1)
            elif tensor.ndim == 4:
                mean = mean.view(1, -1, 1, 1)
                std = std.view(1, -1, 1, 1)
            return tensor * std + mean
        return tensor

    def _spatial_grads(self, f):
        if isinstance(f, torch.Tensor):
            f = f.cpu().detach().numpy()
        original_ndim = f.ndim
        if original_ndim == 2:
            f = f[np.newaxis, ...]
        dy_f = np.zeros_like(f)
        dx_f = np.zeros_like(f)
        dy_f[..., 1:-1, :] = (f[..., 2:, :] - f[..., :-2, :]) / (2 * self.dy)
        dx_f[..., :, 1:-1] = (f[..., :, 2:] - f[..., :, :-2]) / (2 * self.dx)
        dy_f[..., 0, :] = (f[..., 1, :] - f[..., 0, :]) / self.dy
        dy_f[..., -1, :] = (f[..., -1, :] - f[..., -2, :]) / self.dy
        dx_f[..., :, 0] = (f[..., :, 1] - f[..., :, 0]) / self.dx
        dx_f[..., :, -1] = (f[..., :, -1] - f[..., :, -2]) / self.dx
        if original_ndim == 2:
            return dy_f[0], dx_f[0]
        return dy_f, dx_f

    def _laplacian(self, f):
        if isinstance(f, torch.Tensor):
            f = f.cpu().detach().numpy()
        original_ndim = f.ndim
        if original_ndim == 2:
            f = f[np.newaxis, ...]
        lap = np.zeros_like(f)
        lap[..., :, 1:-1] += (f[..., :, 2:] - 2*f[..., :, 1:-1] + f[..., :, :-2]) / (self.dx**2)
        lap[..., 1:-1, :] += (f[..., 2:, :] - 2*f[..., 1:-1, :] + f[..., :-2, :]) / (self.dy**2)
        lap[..., 0, :]  += (f[..., 2, :] - 2*f[..., 1, :] + f[..., 0, :]) / (self.dy**2)
        lap[..., -1, :] += (f[..., -3, :] - 2*f[..., -2, :] + f[..., -1, :]) / (self.dy**2)
        lap[..., :, 0]  += (f[..., :, 2] - 2*f[..., :, 1] + f[..., :, 0]) / (self.dx**2)
        lap[..., :, -1] += (f[..., :, -3] - 2*f[..., :, -2] + f[..., :, -1]) / (self.dx**2)
        if original_ndim == 2:
            return lap[0]
        return lap

    def _temporal_grad(self, f_curr, f_prev):
        if self.dt is None or f_prev is None:
            return 0.0
        if isinstance(f_curr, torch.Tensor):
            f_curr = f_curr.cpu().detach().numpy()
        if isinstance(f_prev, torch.Tensor):
            f_prev = f_prev.cpu().detach().numpy()
        return (f_curr - f_prev) / self.dt

    def _calc_sobolev_split(self, preds, targets, space_order=2, time_order=1,
                            prev_preds=None, prev_targets=None):
        """
        Sobolev calculation: always computes spatial derivatives up to order 2,
        and reports cumulative Sobolev norms at each level:
            sobolev_s0   / rel_sobolev_s0    (order 0 only)
            sobolev_s01  / rel_sobolev_s01   (order 0 + 1)
            sobolev_s012 / rel_sobolev_s012  (order 0 + 1 + 2)
            sobolev_time / rel_sobolev_time  (temporal part)
            sobolev_0    / rel_sobolev_0     (s0  + time)
            sobolev_01   / rel_sobolev_01    (s01 + time)
            sobolev_012  / rel_sobolev_012   (s012 + time)
        """
        max_space_order = 2  # always compute up to 2

        # --- Order 0 (pointwise) ---
        current_p = [preds]
        current_t = [targets]

        order0_err = float(np.mean(np.abs(preds - targets)))
        order0_norm = float(np.mean(np.abs(targets)))

        # --- Spatial derivative orders 1..max_space_order ---
        per_order_err = [order0_err]
        per_order_norm = [order0_norm]

        for k in range(1, max_space_order + 1):
            next_p, next_t = [], []
            order_err = 0.0
            order_norm = 0.0
            for p, t in zip(current_p, current_t):
                dx_p = (p[..., 1:] - p[..., :-1]) / self.dx
                dx_t = (t[..., 1:] - t[..., :-1]) / self.dx
                dy_p = (p[..., 1:, :] - p[..., :-1, :]) / self.dy
                dy_t = (t[..., 1:, :] - t[..., :-1, :]) / self.dy
                order_err += np.mean(np.abs(dx_p - dx_t)) + np.mean(np.abs(dy_p - dy_t))
                order_norm += np.mean(np.abs(dx_t)) + np.mean(np.abs(dy_t))
                next_p.extend([dx_p, dy_p])
                next_t.extend([dx_t, dy_t])
            per_order_err.append(order_err)
            per_order_norm.append(order_norm)
            current_p, current_t = next_p, next_t

        # Cumulative sums for space
        s0_err = per_order_err[0]
        s0_norm = per_order_norm[0]
        s01_err = s0_err + per_order_err[1]
        s01_norm = s0_norm + per_order_norm[1]
        s012_err = s01_err + per_order_err[2]
        s012_norm = s01_norm + per_order_norm[2]

        # --- Temporal derivative (order 1 only) ---
        time_err = 0.0
        time_norm = 0.0
        if time_order >= 1 and prev_preds is not None and prev_targets is not None:
            dt_p = self._temporal_grad(preds, prev_preds)
            dt_t = self._temporal_grad(targets, prev_targets)
            time_err = float(np.mean(np.abs(dt_p - dt_t)))
            time_norm = float(np.mean(np.abs(dt_t)))

        # --- Build result dict ---
        def _rel(e, n):
            return e / (n + 1e-12) if n > 0 else 0.0

        rel_s0 = _rel(s0_err, s0_norm)
        rel_s01 = _rel(s01_err, s01_norm)
        rel_s012 = _rel(s012_err, s012_norm)
        rel_time = _rel(time_err, time_norm)

        return {
            # Space-only cumulative
            "sobolev_s0":   s0_err,       "rel_sobolev_s0":   rel_s0,
            "sobolev_s01":  s01_err,      "rel_sobolev_s01":  rel_s01,
            "sobolev_s012": s012_err,     "rel_sobolev_s012": rel_s012,
            # Time
            "sobolev_time": time_err,     "rel_sobolev_time": rel_time,
            # Combined (space + time)
            "sobolev_0":    s0_err + time_err,
            "rel_sobolev_0":  _rel(s0_err + time_err, s0_norm + time_norm),
            "sobolev_01":   s01_err + time_err,
            "rel_sobolev_01": _rel(s01_err + time_err, s01_norm + time_norm),
            "sobolev_012":  s012_err + time_err,
            "rel_sobolev_012": _rel(s012_err + time_err, s012_norm + time_norm),
        }

    def _calc_h1_abs_rel(self, vars_pred, vars_gt):
        """
        Shared H1 error returning (absolute, relative).
        """
        err = 0.0
        norm = 0.0
        for vp, vgt in zip(vars_pred, vars_gt):
            dy_p, dx_p = self._spatial_grads(vp)
            dy_gt, dx_gt = self._spatial_grads(vgt)
            err += np.mean(np.abs(dx_p - dx_gt)) + np.mean(np.abs(dy_p - dy_gt))
            norm += np.mean(np.abs(dx_gt)) + np.mean(np.abs(dy_gt))
        err /= len(vars_pred)
        norm /= len(vars_pred)
        rel = err / (norm + 1e-12)
        return err, rel


# ========================================================================================
#  IncompressibleLoss  (space_order=2, time_order=1)
# ========================================================================================

class IncompressibleLoss(PhysicalLossBase):
    def __init__(self, preds, labels, constants, Re=2500.0, transpose=False, denorm=True, **kwargs):
        super().__init__(constants, Re=Re, **kwargs)
        self.transpose = transpose
        self.do_denorm = denorm
        self.preds = self._process_data(preds)
        self.u_pred, self.v_pred, self.p_pred = self._extract_uvp(self.preds)
        self.labels = self._process_data(labels)
        self.u_gt, self.v_gt, self.p_gt = self._extract_uvp(self.labels)

    def _process_data(self, data):
        data = self.denormalize(data)
        if isinstance(data, torch.Tensor):
            data = data.cpu().detach().numpy()
        if not self.transpose:
            data = np.swapaxes(data, -2, -1)
        return data

    def _extract_uvp(self, data):
        c_dim = 1 if data.ndim == 4 else 0
        if data.shape[c_dim] == 3:
            u_idx, v_idx, p_idx = 0, 1, 2
        else:
            u_idx, v_idx, p_idx = 1, 2, 3
        if data.ndim == 4:
            u, v, p = data[:, u_idx], data[:, v_idx], data[:, p_idx]
        else:
            u, v, p = data[u_idx], data[v_idx], data[p_idx]
        return u, v, p

    def _calc_continuity(self):
        _, du_dx = self._spatial_grads(self.u_pred)
        dv_dy, _ = self._spatial_grads(self.v_pred)
        div = np.mean(np.abs(du_dx + dv_dy))
        _, du_dx_gt = self._spatial_grads(self.u_gt)
        dv_dy_gt, _ = self._spatial_grads(self.v_gt)
        scale = np.mean(np.abs(du_dx_gt)) + np.mean(np.abs(dv_dy_gt)) + 1e-12
        return div, div / scale

    def _calc_momentum(self):
        du_dy, du_dx = self._spatial_grads(self.u_pred)
        dv_dy, dv_dx = self._spatial_grads(self.v_pred)
        dp_dy, dp_dx = self._spatial_grads(self.p_pred)
        d2u_dy2, _ = self._spatial_grads(du_dy)
        _, d2u_dx2 = self._spatial_grads(du_dx)
        d2v_dy2, _ = self._spatial_grads(dv_dy)
        _, d2v_dx2 = self._spatial_grads(dv_dx)
        visc_u = (d2u_dx2 + d2u_dy2) / self.Re
        visc_v = (d2v_dx2 + d2v_dy2) / self.Re
        res_u = (self.u_pred * du_dx + self.v_pred * du_dy) + dp_dx - visc_u
        res_v = (self.u_pred * dv_dx + self.v_pred * dv_dy) + dp_dy - visc_v
        abs_val = np.mean(np.abs(res_u) + np.abs(res_v))
        term_scale = (np.mean(np.abs(self.u_pred * du_dx)) + np.mean(np.abs(self.v_pred * du_dy))
                      + np.mean(np.abs(dp_dx)) + np.mean(np.abs(visc_u))
                      + np.mean(np.abs(self.u_pred * dv_dx)) + np.mean(np.abs(self.v_pred * dv_dy))
                      + np.mean(np.abs(dp_dy)) + np.mean(np.abs(visc_v)) + 1e-12)
        return abs_val, abs_val / term_scale

    def _calc_vorticity_error(self):
        if self.u_gt is None:
            return 0.0, 0.0
        _, dv_dx = self._spatial_grads(self.v_pred)
        du_dy, _ = self._spatial_grads(self.u_pred)
        curl_pred = dv_dx - du_dy
        _, dv_dx_gt = self._spatial_grads(self.v_gt)
        du_dy_gt, _ = self._spatial_grads(self.u_gt)
        curl_gt = dv_dx_gt - du_dy_gt
        abs_val = np.mean(np.abs(curl_pred - curl_gt))
        rel_val = abs_val / (np.mean(np.abs(curl_gt)) + 1e-12)
        return abs_val, rel_val

    def _calc_h1_error(self):
        if self.u_gt is None:
            return 0.0, 0.0
        return self._calc_h1_abs_rel(
            [self.u_pred, self.v_pred],
            [self.u_gt, self.v_gt]
        )

    def _calc_sobolev_metric(self):
        """NS: space_order=2 (unified), time_order=1 (no prev data for incompressible)."""
        if self.labels is None:
            return {k: 0.0 for k in [
                "sobolev_s0", "rel_sobolev_s0", "sobolev_s01", "rel_sobolev_s01",
                "sobolev_s012", "rel_sobolev_s012", "sobolev_time", "rel_sobolev_time",
                "sobolev_0", "rel_sobolev_0", "sobolev_01", "rel_sobolev_01",
                "sobolev_012", "rel_sobolev_012"]}
        return self._calc_sobolev_split(
            self.preds, self.labels,
            space_order=2, time_order=1,
            prev_preds=None, prev_targets=None
        )

    def compute(self):
        cont, rel_cont = self._calc_continuity()
        mom, rel_mom = self._calc_momentum()
        vort, rel_vort = self._calc_vorticity_error()
        h1, rel_h1 = self._calc_h1_error()
        sob_dict = self._calc_sobolev_metric()
        result = {
            "continuity": cont,      "rel_continuity": rel_cont,
            "momentum": mom,         "rel_momentum": rel_mom,
            "vorticity_err": vort,   "rel_vorticity_err": rel_vort,
            "h1_error": h1,          "rel_h1_error": rel_h1,
        }
        result.update(sob_dict)
        return result


# ========================================================================================
#  CompressibleLoss  (space_order=1, time_order=1)
# ========================================================================================

class CompressibleLoss(PhysicalLossBase):
    def __init__(self, preds, labels, prev_preds, prev_labels, constants,
                 gamma=1.4, mean_pressure=0.0, transpose=False, denorm=True,
                 dt=1.0, has_gravity=False, **kwargs):
        super().__init__(constants, dt=dt, **kwargs)
        self.gamma = gamma
        self.mean_pressure = mean_pressure
        self.transpose = transpose
        self.do_denorm = denorm
        self.has_gravity = has_gravity
        self.n_base_channels = 4

        if prev_preds is None or prev_labels is None:
            raise ValueError("CompressibleLoss needs prev_preds and prev_labels!")

        self.preds_raw = self._process_data(preds)
        self.rho, self.u, self.v, self.p, self.E, self.g = self._extract_vars(self.preds_raw)
        self.prev_preds_raw = self._process_data(prev_preds)
        self.rho_prev, self.u_prev, self.v_prev, self.p_prev, self.E_prev, self.g_prev = \
            self._extract_vars(self.prev_preds_raw)
        if labels is not None:
            self.labels_raw = self._process_data(labels)
            self.rho_gt, self.u_gt, self.v_gt, self.p_gt, _, self.g_gt = \
                self._extract_vars(self.labels_raw)
        else:
            self.labels_raw = None
        if prev_labels is not None:
            self.prev_labels_raw = self._process_data(prev_labels)
        else:
            self.prev_labels_raw = None

    def _process_data(self, data):
        if self.do_denorm:
            data = self.denormalize(data)
        if isinstance(data, torch.Tensor):
            data = data.cpu().detach().numpy()
        if not self.transpose:
            data = np.swapaxes(data, -2, -1)
        return data

    def _extract_vars(self, data):
        is_batch = (data.ndim == 4)
        c_dim = 1 if is_batch else 0
        n_channels = data.shape[c_dim]
        if is_batch:
            rho, u, v, p = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
            g = data[:, 4] if n_channels >= 5 else None
        else:
            rho, u, v, p = data[0], data[1], data[2], data[3]
            g = data[4] if n_channels >= 5 else None
        if self.mean_pressure != 0.0:
            p = p + self.mean_pressure
        kin_energy = 0.5 * rho * (u**2 + v**2)
        E = p / (self.gamma - 1.0) + kin_energy
        return rho, u, v, p, E, g

    def _calc_mass_cons(self):
        _, dfx_dx = self._spatial_grads(self.rho * self.u)
        dfy_dy, _ = self._spatial_grads(self.rho * self.v)
        drho_dt = self._temporal_grad(self.rho, self.rho_prev)
        residual = np.mean(np.abs(drho_dt + dfx_dx + dfy_dy))
        scale = (np.mean(np.abs(drho_dt)) + np.mean(np.abs(dfx_dx))
                 + np.mean(np.abs(dfy_dy)) + 1e-12)
        return residual, residual / scale

    def _calc_momentum_cons(self):
        mom_x = self.rho * self.u
        _, dfx_mx = self._spatial_grads(self.rho * self.u**2 + self.p)
        dfy_mx, _ = self._spatial_grads(self.rho * self.u * self.v)
        dmx_dt = self._temporal_grad(mom_x, self.rho_prev * self.u_prev)
        res_mx = np.mean(np.abs(dmx_dt + dfx_mx + dfy_mx))
        scale_mx = np.mean(np.abs(dmx_dt)) + np.mean(np.abs(dfx_mx)) + np.mean(np.abs(dfy_mx)) + 1e-12
        mom_y = self.rho * self.v
        _, dfx_my = self._spatial_grads(self.rho * self.u * self.v)
        dfy_my, _ = self._spatial_grads(self.rho * self.v**2 + self.p)
        dmy_dt = self._temporal_grad(mom_y, self.rho_prev * self.v_prev)
        rhs_y = 0.0
        if self.has_gravity and self.g is not None:
            rhs_y = self.rho * self.g
        res_my = np.mean(np.abs(dmy_dt + dfx_my + dfy_my - rhs_y))
        scale_my = np.mean(np.abs(dmy_dt)) + np.mean(np.abs(dfx_my)) + np.mean(np.abs(dfy_my)) + 1e-12

        abs_val = (res_mx + res_my) / 2.0
        rel_val = (res_mx / scale_mx + res_my / scale_my) / 2.0
        return abs_val, rel_val

    def _calc_energy_cons(self):
        _, dfx_E = self._spatial_grads((self.E + self.p) * self.u)
        dfy_E, _ = self._spatial_grads((self.E + self.p) * self.v)
        dE_dt = self._temporal_grad(self.E, self.E_prev)
        rhs = 0.0
        if self.has_gravity and self.g is not None:
            rhs = self.rho * self.v * self.g
        residual = np.mean(np.abs(dE_dt + dfx_E + dfy_E - rhs))
        scale = (np.mean(np.abs(dE_dt)) + np.mean(np.abs(dfx_E))
                 + np.mean(np.abs(dfy_E)) + 1e-12)
        return residual, residual / scale

    def _calc_h1_error(self):
        if self.labels_raw is None:
            return 0.0, 0.0
        vars_pred = [self.rho, self.u, self.v, self.p]
        vars_gt = [self.rho_gt, self.u_gt, self.v_gt, self.p_gt]
        if self.has_gravity and self.g is not None and self.g_gt is not None:
            vars_pred.append(self.g)
            vars_gt.append(self.g_gt)
        return self._calc_h1_abs_rel(vars_pred, vars_gt)

    def _calc_sobolev_metric(self):
        """CE: space_order=2 (unified), time_order=1."""
        if self.labels_raw is None:
            return {k: 0.0 for k in [
                "sobolev_s0", "rel_sobolev_s0", "sobolev_s01", "rel_sobolev_s01",
                "sobolev_s012", "rel_sobolev_s012", "sobolev_time", "rel_sobolev_time",
                "sobolev_0", "rel_sobolev_0", "sobolev_01", "rel_sobolev_01",
                "sobolev_012", "rel_sobolev_012"]}
        return self._calc_sobolev_split(
            self.preds_raw, self.labels_raw,
            space_order=2, time_order=1,
            prev_preds=self.prev_preds_raw, prev_targets=self.prev_labels_raw
        )

    def compute(self):
        mass, rel_mass = self._calc_mass_cons()
        mom, rel_mom = self._calc_momentum_cons()
        energy, rel_energy = self._calc_energy_cons()
        h1, rel_h1 = self._calc_h1_error()
        sob_dict = self._calc_sobolev_metric()
        result = {
            "mass_cons": mass,       "rel_mass_cons": rel_mass,
            "momentum_cons": mom,    "rel_momentum_cons": rel_mom,
            "energy_cons": energy,   "rel_energy_cons": rel_energy,
            "h1_error": h1,          "rel_h1_error": rel_h1,
        }
        result.update(sob_dict)
        return result


# ========================================================================================
#  WaveLoss  (space_order=1, time_order=1, only speed_cons as physics metric)
# ========================================================================================

class WaveLoss:
    def __init__(self, preds, labels, prev_preds, prev_labels, constants,
                 prev_prev_preds=None, prev_prev_labels=None,
                 transpose=False, denorm=True, dt=None,
                 dx=1.0/128, dy=1.0/128):
        self.constants = constants
        self.transpose = transpose
        self.do_denorm = denorm
        self.dx = dx
        self.dy = dy
        if dt is None:
            time_step_size = constants.get("time_step_size", 2)
            total_time = constants.get("time", 1.0)
            self.dt = time_step_size / total_time
        else:
            self.dt = dt

        if prev_preds is None or prev_labels is None:
            raise ValueError("Wave PDE needs prev_preds and prev_labels!")

        self.has_three_steps = (prev_prev_preds is not None)

        self.preds_raw = self._process_data(preds)
        self.u, self.c = self._extract_vars(self.preds_raw)
        self.prev_preds_raw = self._process_data(prev_preds)
        self.u_prev, self.c_prev = self._extract_vars(self.prev_preds_raw)

        if prev_prev_preds is not None:
            self.pp_preds_raw = self._process_data(prev_prev_preds)
            self.u_pp, self.c_pp = self._extract_vars(self.pp_preds_raw)
        else:
            self.pp_preds_raw = None
            self.u_pp = self.c_pp = None

        if labels is not None:
            self.labels_raw = self._process_data(labels)
            self.u_gt, self.c_gt = self._extract_vars(self.labels_raw)
        else:
            self.labels_raw = None
            self.u_gt = self.c_gt = None

        if prev_labels is not None:
            self.prev_labels_raw = self._process_data(prev_labels)
            self.u_prev_gt, self.c_prev_gt = self._extract_vars(self.prev_labels_raw)
        else:
            self.prev_labels_raw = None

        if prev_prev_labels is not None:
            self.pp_labels_raw = self._process_data(prev_prev_labels)
        else:
            self.pp_labels_raw = None

    def denormalize(self, tensor):
        if self.constants is None:
            return tensor
        mean_u = self.constants["mean"]
        std_u  = self.constants["std"]
        mean_c = self.constants["mean_c"]
        std_c  = self.constants["std_c"]
        if isinstance(tensor, np.ndarray):
            out = tensor.copy()
            if tensor.ndim == 4:
                out[:, 0] = tensor[:, 0] * std_u + mean_u
                if tensor.shape[1] >= 2:
                    out[:, 1] = tensor[:, 1] * std_c + mean_c
            elif tensor.ndim == 3:
                out[0] = tensor[0] * std_u + mean_u
                if tensor.shape[0] >= 2:
                    out[1] = tensor[1] * std_c + mean_c
            return out
        elif isinstance(tensor, torch.Tensor):
            out = tensor.clone()
            if tensor.ndim == 4:
                out[:, 0] = tensor[:, 0] * std_u + mean_u
                if tensor.shape[1] >= 2:
                    out[:, 1] = tensor[:, 1] * std_c + mean_c
            elif tensor.ndim == 3:
                out[0] = tensor[0] * std_u + mean_u
                if tensor.shape[0] >= 2:
                    out[1] = tensor[1] * std_c + mean_c
            return out
        return tensor

    def _process_data(self, data):
        if self.do_denorm:
            data = self.denormalize(data)
        if isinstance(data, torch.Tensor):
            data = data.cpu().detach().numpy()
        if not self.transpose:
            data = np.swapaxes(data, -2, -1)
        return data

    def _extract_vars(self, data):
        if data.ndim == 4:
            u = data[:, 0]
            c = data[:, 1] if data.shape[1] >= 2 else None
        else:
            u = data[0]
            c = data[1] if data.shape[0] >= 2 else None
        return u, c

    def _spatial_grads(self, f):
        if isinstance(f, torch.Tensor):
            f = f.cpu().detach().numpy()
        original_ndim = f.ndim
        if original_ndim == 2:
            f = f[np.newaxis, ...]
        dy_f = np.zeros_like(f)
        dx_f = np.zeros_like(f)
        dy_f[..., 1:-1, :] = (f[..., 2:, :] - f[..., :-2, :]) / (2 * self.dy)
        dx_f[..., :, 1:-1] = (f[..., :, 2:] - f[..., :, :-2]) / (2 * self.dx)
        dy_f[..., 0, :]  = (f[..., 1, :]  - f[..., 0, :])  / self.dy
        dy_f[..., -1, :] = (f[..., -1, :] - f[..., -2, :]) / self.dy
        dx_f[..., :, 0]  = (f[..., :, 1]  - f[..., :, 0])  / self.dx
        dx_f[..., :, -1] = (f[..., :, -1] - f[..., :, -2]) / self.dx
        if original_ndim == 2:
            return dy_f[0], dx_f[0]
        return dy_f, dx_f

    def _temporal_grad(self, f_curr, f_prev):
        if isinstance(f_curr, torch.Tensor):
            f_curr = f_curr.cpu().detach().numpy()
        if isinstance(f_prev, torch.Tensor):
            f_prev = f_prev.cpu().detach().numpy()
        return (f_curr - f_prev) / self.dt

    def _calc_speed_cons(self):
        if self.c is None or self.c_prev is None:
            return 0.0, 0.0
        abs_val = np.mean(np.abs(self.c - self.c_prev))
        rel_val = abs_val / (np.mean(np.abs(self.c_prev)) + 1e-12)
        return abs_val, rel_val

    def _calc_sobolev_metric(self):
        """Wave: space_order=2 (unified), time_order=1."""
        zero_keys = [
            "sobolev_s0", "rel_sobolev_s0", "sobolev_s01", "rel_sobolev_s01",
            "sobolev_s012", "rel_sobolev_s012", "sobolev_time", "rel_sobolev_time",
            "sobolev_0", "rel_sobolev_0", "sobolev_01", "rel_sobolev_01",
            "sobolev_012", "rel_sobolev_012"]
        if self.labels_raw is None:
            return {k: 0.0 for k in zero_keys}

        preds = self.preds_raw
        targets = self.labels_raw
        max_space_order = 2

        current_p = [preds]
        current_t = [targets]

        # Order 0
        order0_err = float(np.mean(np.abs(preds - targets)))
        order0_norm = float(np.mean(np.abs(targets)))

        per_order_err = [order0_err]
        per_order_norm = [order0_norm]

        for k in range(1, max_space_order + 1):
            next_p, next_t = [], []
            order_err = 0.0
            order_norm = 0.0
            for p, t in zip(current_p, current_t):
                dx_p = (p[..., 1:] - p[..., :-1]) / self.dx
                dx_t = (t[..., 1:] - t[..., :-1]) / self.dx
                dy_p = (p[..., 1:, :] - p[..., :-1, :]) / self.dy
                dy_t = (t[..., 1:, :] - t[..., :-1, :]) / self.dy
                order_err += np.mean(np.abs(dx_p - dx_t)) + np.mean(np.abs(dy_p - dy_t))
                order_norm += np.mean(np.abs(dx_t)) + np.mean(np.abs(dy_t))
                next_p.extend([dx_p, dy_p])
                next_t.extend([dx_t, dy_t])
            per_order_err.append(order_err)
            per_order_norm.append(order_norm)
            current_p, current_t = next_p, next_t

        # Cumulative sums for space
        s0_err = per_order_err[0]
        s0_norm = per_order_norm[0]
        s01_err = s0_err + per_order_err[1]
        s01_norm = s0_norm + per_order_norm[1]
        s012_err = s01_err + per_order_err[2]
        s012_norm = s01_norm + per_order_norm[2]

        # Temporal
        time_err = 0.0
        time_norm = 0.0
        if self.prev_labels_raw is not None and self.prev_preds_raw is not None:
            dt_p = self._temporal_grad(preds, self.prev_preds_raw)
            dt_t = self._temporal_grad(targets, self.prev_labels_raw)
            time_err = float(np.mean(np.abs(dt_p - dt_t)))
            time_norm = float(np.mean(np.abs(dt_t)))

        def _rel(e, n):
            return e / (n + 1e-12) if n > 0 else 0.0

        return {
            "sobolev_s0":   s0_err,       "rel_sobolev_s0":   _rel(s0_err, s0_norm),
            "sobolev_s01":  s01_err,      "rel_sobolev_s01":  _rel(s01_err, s01_norm),
            "sobolev_s012": s012_err,     "rel_sobolev_s012": _rel(s012_err, s012_norm),
            "sobolev_time": time_err,     "rel_sobolev_time": _rel(time_err, time_norm),
            "sobolev_0":    s0_err + time_err,
            "rel_sobolev_0":  _rel(s0_err + time_err, s0_norm + time_norm),
            "sobolev_01":   s01_err + time_err,
            "rel_sobolev_01": _rel(s01_err + time_err, s01_norm + time_norm),
            "sobolev_012":  s012_err + time_err,
            "rel_sobolev_012": _rel(s012_err + time_err, s012_norm + time_norm),
        }

    def compute(self):
        """Wave: only speed_cons + sobolev (space/time/sum)."""
        speed, rel_speed = self._calc_speed_cons()
        sob_dict = self._calc_sobolev_metric()
        result = {
            "speed_cons": speed,     "rel_speed_cons": rel_speed,
        }
        result.update(sob_dict)
        return result