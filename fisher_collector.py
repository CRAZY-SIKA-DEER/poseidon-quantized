import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import torch.nn.functional as F
from MySVD.helper.whitener import *
from tqdm import tqdm
from poseidon_main.scOT.problems.fluids.normalization_constants import CONSTANTS

def sobolev_loss_calc_denorm(dataset_name, pred, labels, order_control, base_loss_val, 
                            loss_func, device, constants,
                            prev_pred=None,
                            prev_labels=None,
                            transpose=False,
                            decay_alpha=1, balance_ratio=0.5):
    """
    修复版: 强制将 labels, mean, std 全部移动到 device，防止 device mismatch 报错。
    """
    # [修复 1] 强制将 labels 搬运到 GPU (如果它还在 CPU)
    # pred 通常已经在 GPU 上，但为了保险也可以 .to(device)
    if labels.device != device:
        labels = labels.to(device)
    
    # 1. 准备统计量 (Mean/Std)
    # [修复 2] 确保 mean/std 创建时就在 GPU 上
    if not torch.is_tensor(constants['mean']):
        mean = torch.tensor(constants['mean'], device=device)
    else:
        mean = constants['mean'].to(device)
        
    if not torch.is_tensor(constants['std']):
        std = torch.tensor(constants['std'], device=device)
    else:
        std = constants['std'].to(device)

    # 调整维度以便广播
    if mean.ndim == 1: mean = mean.view(1, -1, 1, 1)
    if std.ndim == 1: std = std.view(1, -1, 1, 1)

    # 2. 反归一化 (还原物理量级)
    # 此时所有变量都在 device 上，不会报错
    pred_phys = pred * std + mean
    labels_phys = labels * std + mean
    time_loss = 0.0
    if prev_pred is not None and prev_labels is not None:
        prev_pred_phys = prev_pred * std + mean
        prev_labels_phys = prev_labels * std + mean
        pred_time_diff = pred_phys - prev_pred_phys
        labels_time_diff = labels_phys - prev_labels_phys
        time_loss = loss_func(pred_time_diff, labels_time_diff)

    C = pred_phys.shape[1]
    if "incompressible" in dataset_name:
        if C == 3:
            indices = [0, 1, 2] # [u, v, p]
        else:
            indices = [1, 2, 3] # [rho, u, v, p]
    elif "compressible" in dataset_name:
        indices = [0, 1, 2, 3]
    elif "wave" in dataset_name.lower() or "Wave" in dataset_name:
        indices = list(range(C)) 
        
    if not transpose:
        pred_phys =  pred_phys.transpose(-2, -1)
        labels_phys =  labels_phys.transpose(-2, -1)
    # [修复 3] 确保 pred_uv 和 target_uv 是连续的内存 (可选，但推荐)
    pred_uv = pred_phys[:, indices, ...].contiguous()
    target_uv = labels_phys[:, indices, ...].contiguous()

    # 初始化循环变量
    current_preds = [pred_uv]
    current_targets = [target_uv]
    total_weighted_grad_loss = 0.0

    # Order 0: 函数值本身的 loss (反归一化物理空间)
    order0_loss = loss_func(pred_uv, target_uv)
    order0_loss_val = order0_loss.item() if isinstance(order0_loss, torch.Tensor) else order0_loss
    if order0_loss_val > 1e-20:
        norm_weight_0 = (base_loss_val / order0_loss_val) * balance_ratio
    else:
        norm_weight_0 = 0.0
    norm_weight_0 = min(norm_weight_0, 1000.0)
    total_weighted_grad_loss += norm_weight_0 * order0_loss

    # 4. 循环计算导数
    for k in range(1, order_control + 1):
        next_preds = []
        next_targets = []
        current_order_loss = 0.0 
        
        for p, t in zip(current_preds, current_targets):
            # 计算梯度
            # 这里的切片操作后的 tensor 依然在 device 上
            dx_p = (p[..., 1:] - p[..., :-1])
            dx_t = (t[..., 1:] - t[..., :-1])
            dy_p = (p[..., 1:, :] - p[..., :-1, :])
            dy_t = (t[..., 1:, :] - t[..., :-1, :])
            
            # 累加 Loss
            current_order_loss += loss_func(dx_p, dx_t) + loss_func(dy_p, dy_t)
            
            if k < order_control:
                next_preds.extend([dx_p, dy_p])
                next_targets.extend([dx_t, dy_t])
        
        order_loss_val = current_order_loss.item() if isinstance(current_order_loss, torch.Tensor) else current_order_loss

        if order_loss_val > 1e-20:
            norm_weight = (base_loss_val / order_loss_val) * balance_ratio
        else:
            norm_weight = 0.0
        
        # 限制权重上限
        norm_weight = min(norm_weight, 1000.0) 
        
        decay = decay_alpha ** (k - 1)
        final_lambda_k = norm_weight * decay
        
        total_weighted_grad_loss += final_lambda_k * current_order_loss
        
        current_preds = next_preds
        current_targets = next_targets

    if isinstance(time_loss, torch.Tensor):
        time_loss_val = time_loss.item()
        if time_loss_val > 1e-20:
            time_weight = (base_loss_val / time_loss_val) * balance_ratio 
        else:
            time_weight = 0.0
        time_weight = min(time_weight, 1000.0)
        total_weighted_grad_loss += time_weight * time_loss

    return total_weighted_grad_loss


def sobolev_loss_calc_explicit(pred, labels, order_control, base_loss_val, loss_func, device, constants, transpose=False, decay_alpha=1, balance_ratio=1):
    """
    修改版: 显式拆解了 u, v 在 x, y 方向的每一个导数分量。
    方便通过注释代码来剔除 非对角项 (Cross-terms) 或 调试特定方向。
    """
    pred = pred.to(device)
    labels = labels.to(device)
    mean = torch.tensor(constants['mean'], device=device).view(1, -1, 1, 1)
    std = torch.tensor(constants['std'], device=device).view(1, -1, 1, 1)
    pred = pred * std + mean
    labels = labels * std + mean
    C = pred.shape[1]
    if C == 3:
        u_idx, v_idx, p_idx = 0, 1, 2
    else:
        u_idx, v_idx, p_idx = 1, 2, 3
    # 调整维度以便广播
    if mean.ndim == 1: mean = mean.view(1, -1, 1, 1)
    if std.ndim == 1: std = std.view(1, -1, 1, 1)
    # 1. 拆分通道 (假设前两个通道是 u 和 v)
    # pred shape: [B, C, H, W] -> u, v shape: [B, H, W]

    u_pred = pred[:, u_idx, ...]
    v_pred = pred[:, v_idx, ...]
    p_pred = pred[:, p_idx, ...]
    u_label = labels[:, u_idx, ...]
    v_label = labels[:, v_idx, ...]
    p_label = labels[:, p_idx, ...]
    if not transpose:
        u_pred = u_pred.transpose(-2, -1)
        v_pred = v_pred.transpose(-2, -1)
        p_pred = p_pred.transpose(-2, -1)

        u_label = u_label.transpose(-2, -1)
        v_label = v_label.transpose(-2, -1)  
        p_label = p_label.transpose(-2, -1)    

    # 2. 定义差分算子 (Helper)
    def diff_x(a): return (a[..., 1:] - a[..., :-1]).to(device)
    def diff_y(a): return (a[..., 1:, :] - a[..., :-1, :]).to(device)

    total_weighted_grad_loss = 0.0

    # =========================================================
    # Order 0: 函数值本身的 loss (反归一化物理空间)
    # =========================================================
    order0_loss = (loss_func(u_pred, u_label) + 
                   loss_func(v_pred, v_label) + 
                   loss_func(p_pred, p_label))
    total_weighted_grad_loss += order0_loss

    # =========================================================
    # Order 1: 一阶导数 (最关键的部分)
    # =========================================================
    if order_control >= 1:
        # --- A. 计算四个分量的导数 ---
        # u
        du_dx_p = diff_x(u_pred)
        du_dx_t = diff_x(u_label)
        du_dy_p = diff_y(u_pred)
        du_dy_t = diff_y(u_label)
        # v
        dv_dy_p = diff_y(v_pred)
        dv_dy_t = diff_y(v_label)
        dv_dx_p = diff_x(v_pred)
        dv_dx_t = diff_x(v_label)
        # P
        dp_dx_p = diff_x(p_pred)
        dp_dx_t = diff_x(p_label)
        dp_dy_p = diff_y(p_pred)
        dp_dy_t = diff_y(p_label)

        # 4. v 对 x (非对角项 - 属于 Vorticity/Shear)
        du_dx_p_cut = du_dx_p[..., :-1, :]
        du_dx_t_cut = du_dx_t[..., :-1, :]
        dv_dy_p_cut = dv_dy_p[..., :, :-1]
        dv_dy_t_cut = dv_dy_t[..., :, :-1]
        # --- B. 分别计算 Loss ---
        l_du_dx = loss_func(du_dx_p, du_dx_t)
        l_dv_dy = loss_func(dv_dy_p, dv_dy_t)
        l_du_dy = loss_func(du_dy_p, du_dy_t)
        l_dv_dx = loss_func(dv_dx_p, dv_dx_t)
        l_dp_dx = loss_func(dp_dx_p, dp_dx_t)
        l_dp_dy = loss_func(dp_dy_p, dp_dy_t)


        div_pred = du_dx_p_cut + dv_dy_p_cut
        div_gt = du_dx_t_cut + dv_dy_t_cut
        l_div = loss_func(div_pred, div_gt)
        # --- C. 【关键】在这里注释/取消注释来选择分量 ---
        loss_terms_order1 = []
        

        # Divergence
        # loss_terms_order1.append(l_div)

        # Sobolev
        loss_terms_order1.append(l_du_dx) 
        loss_terms_order1.append(l_dv_dy) 
        loss_terms_order1.append(l_du_dy) 
        loss_terms_order1.append(l_dv_dx) 
        loss_terms_order1.append(l_dp_dx)
        loss_terms_order1.append(l_dp_dy)
        # 汇总一阶 Loss
        total_weighted_grad_loss = sum(loss_terms_order1)
        # weight = base_loss_val / total_weighted_grad_loss.item()
        total_weighted_grad_loss = 10000 * total_weighted_grad_loss # 100
        # # --- D. 自动权重平衡 (Auto-Balancing) ---
        # order_loss_val = current_order_loss.item() if isinstance(current_order_loss, torch.Tensor) else current_order_loss
        
        # if order_loss_val > 1e-20:
        #     norm_weight = (base_loss_val / order_loss_val) * balance_ratio
        # else:
        #     norm_weight = 0.0
        # norm_weight = min(norm_weight, 100.0) # 限制权重上限
        
        # # 累加到总 Loss
        # total_weighted_grad_loss += norm_weight * current_order_loss

    return total_weighted_grad_loss


def collect_fisher_dict_linear_only(trainer, dataset_name, calib_loader, 
                                    device, gradient_loss="mse",
                                    care_I=False, 
                                    order_control=0, constants=None,
                                    transpose=False, balance_ratio=0.5):
    """
    Collect Fisher Information Matrix for Linear Layers.
    Includes fixes for:
    1. Automatic dimension matching (4D inputs).
    2. Disabling in-place operations to prevent BackwardHook errors.
    """
    if gradient_loss == 'l1':
        loss_func = F.l1_loss
    elif gradient_loss == 'mse':
        loss_func = F.mse_loss
    elif gradient_loss == 'smooth_l1':
        loss_func = F.smooth_l1_loss
    else:
        raise ValueError(f"Unsupported gradient_loss type: {gradient_loss}")

    print("Collecting Fisher Information Matrix (Linear Layers Only)...")
    
    # 1. Prepare Model
    model = trainer.model.to(device).train() # Must be in train mode for backward
    model.zero_grad()
    
    # =========================================================
    # [Fix] Disable In-place Operations
    # This prevents "Output 0 of BackwardHookFunctionBackward is a view..." error
    # caused by ReLU(inplace=True) modifying outputs needed by hooks.
    # =========================================================
    print("Disabling in-place operations for gradient collection...")
    def disable_inplace(m):
        if hasattr(m, 'inplace'):
            m.inplace = False
    
    model.apply(disable_inplace)
    
    cov_dict = {}   
    count_dict = {} 
    
    # --- Hook Definition ---
    def get_fisher_hook(name):
        def hook(module, grad_input, grad_output):
            # grad_output[0] is the gradient w.r.t layer output
            g = grad_output[0].detach()
            
            # Flatten all dimensions (Batch, Time, H, W) -> [N, Out_Dim]
            if g.dim() > 2:
                g = g.reshape(-1, g.shape[-1]) 
            
            # Calculate g^T * g (Approximated Fisher)
            cov = g.t() @ g 
            
            if name not in cov_dict:
                cov_dict[name] = cov
                count_dict[name] = g.shape[0]
            else:
                cov_dict[name] += cov
                count_dict[name] += g.shape[0]
        return hook

    # --- Register Hooks ---
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_full_backward_hook(get_fisher_hook(name)))
            
    # --- Run Data Loop ---    
    for i, batch in enumerate(tqdm(calib_loader)):
        inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        
        # 2. Fix Dimensions (Auto-fix for 3D/5D inputs)
        if "pixel_values" in inputs:
            pv = inputs["pixel_values"]
            if pv.dim() == 3: # [C, H, W] -> [1, C, H, W]
                inputs["pixel_values"] = pv.unsqueeze(0)
            elif pv.dim() == 5: # [B, T, C, H, W] -> [B, C, H, W] (Take 1st frame)
                inputs["pixel_values"] = pv[:, 0, ...]
        # 3. Forward
        try:
            outputs = model(**inputs)
        except RuntimeError as e:
            print(f"[Error] Batch {i} failed. Input shape: {inputs.get('pixel_values', 'N/A').shape}")
            raise e
        # 4. Get Loss
        loss = None
        if isinstance(outputs, dict) and "loss" in outputs:
            loss = outputs["loss"]
        elif hasattr(outputs, "loss"):
            loss = outputs.loss
        elif isinstance(outputs, (list, tuple)):
            loss = outputs[0]
        if loss is None:
            print("loss is none")
            continue
        base_loss_val = loss.item()
        # ==== Calculate Gradient Loss ====
        labels = batch["labels"]
        pred = outputs.output
        # grad_loss = sobolev_loss_calc(pred, labels, order_control, base_loss_val, loss_func, device)
        grad_loss = sobolev_loss_calc_denorm(dataset_name, pred, labels, order_control, base_loss_val, loss_func, device, constants=constants, transpose=transpose, balance_ratio=balance_ratio)
        # ==================================
        loss = loss + grad_loss
        # 5. Backward
        model.zero_grad()
        loss.backward()
        if i == 0: # 只看第一个 batch
            print(f"Current Batch Loss: {loss.item()}")
            for n, p in model.named_parameters():
                if "layers.0.blocks.0.attention.self.query" in n and p.grad is not None:
                    print(f"Gradient Norm for {n}: {p.grad.norm().item()}")
        del outputs, loss

    # --- Post-process (Cholesky) ---
    Ldict = {}
    print("Computing Cholesky decomposition with Trace Normalization...")
    for name, cov in cov_dict.items():
        try:
            N = count_dict[name]
            Sigma = (cov / N).double()         
            dim = Sigma.shape[0]
            diag_mean = torch.diagonal(Sigma).mean().item()
            if care_I:
                if diag_mean > 0:
                    scale = 1.0 / diag_mean
                else:
                    scale = 1.0
                Sigma_final = (Sigma * scale) + torch.eye(dim, device=device, dtype=torch.float64)
                
            else:
                if diag_mean > 0:
                    jitter_val = max(diag_mean * 0.01, 1e-18) 
                else:
                    jitter_val = 1e-9
                
                Sigma_final = Sigma + torch.eye(dim, device=device, dtype=torch.float64) * jitter_val
            try:
                L_mat = torch.linalg.cholesky(Sigma_final)
            except RuntimeError:
                print(f"[Warning] Cholesky failed for {name}. Switching to Eigen...")
                eigenvalues, eigenvectors = torch.linalg.eigh(Sigma_final)
                min_eig = 1e-18 
                eigenvalues = torch.clamp(eigenvalues, min=min_eig)
                S_sqrt = torch.sqrt(eigenvalues)
                L_mat = eigenvectors * S_sqrt.unsqueeze(0)
            L_mat = L_mat.float()
            Lt = L_mat.t() 
            try:
                Lt_inv = torch.linalg.inv(Lt)
            except:
                Lt_inv = torch.linalg.pinv(Lt) 
            Ldict[name] = (Lt.cpu(), Lt_inv.cpu())
            del Sigma, Sigma_final, L_mat, Lt, Lt_inv
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"[Error] Processing {name}: {e}")
            dim = cov.shape[0]
            I_cpu = torch.eye(dim)
            Ldict[name] = (I_cpu, I_cpu)

    for h in handles: h.remove()
    model.eval()
    
    return Ldict



def collect_fisher_dict_time_sob(trainer, dataset_name, calib_loader, prev_loader, 
                                    device, gradient_loss="mse",
                                    care_I=False, 
                                    order_control=0, constants=None,
                                    transpose=False, balance_ratio=0.5):
    """
    Collect Fisher Information Matrix for Linear Layers.
    Includes fixes for:
    1. Automatic dimension matching (4D inputs).
    2. Disabling in-place operations to prevent BackwardHook errors.
    3. [NEW] Dual-dataloader iteration for temporal difference calculations.
    """
    if gradient_loss == 'l1':
        loss_func = F.l1_loss
    elif gradient_loss == 'mse':
        loss_func = F.mse_loss
    elif gradient_loss == 'smooth_l1':
        loss_func = F.smooth_l1_loss
    else:
        raise ValueError(f"Unsupported gradient_loss type: {gradient_loss}")

    print("Collecting Fisher Information Matrix (Linear Layers Only)...")
    
    # 1. Prepare Model
    model = trainer.model.to(device).train() # Must be in train mode for backward
    model.zero_grad()
    
    # =========================================================
    # [Fix] Disable In-place Operations
    # =========================================================
    print("Disabling in-place operations for gradient collection...")
    def disable_inplace(m):
        if hasattr(m, 'inplace'):
            m.inplace = False
    
    model.apply(disable_inplace)
    
    cov_dict = {}   
    count_dict = {} 
    
    # --- Hook Definition ---
    def get_fisher_hook(name):
        def hook(module, grad_input, grad_output):
            g = grad_output[0].detach()
            if g.dim() > 2:
                g = g.reshape(-1, g.shape[-1]) 
            cov = g.t() @ g 
            
            if name not in cov_dict:
                cov_dict[name] = cov
                count_dict[name] = g.shape[0]
            else:
                cov_dict[name] += cov
                count_dict[name] += g.shape[0]
        return hook

    # --- Register Hooks ---
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_full_backward_hook(get_fisher_hook(name)))
            
    # --- [NEW] Run Dual Data Loop ---    
    # 使用 zip 同时遍历当前时间步和上一个时间步的 loader
    for i, (curr_batch, prev_batch) in enumerate(tqdm(zip(calib_loader, prev_loader), total=len(calib_loader))):
        
        # [NEW] 分别准备当前步和上一步的输入
        curr_inputs = {k: v.to(device) for k, v in curr_batch.items() if isinstance(v, torch.Tensor)}
        prev_inputs = {k: v.to(device) for k, v in prev_batch.items() if isinstance(v, torch.Tensor)}
        
        # [NEW] 修复两个输入的维度
        for inputs in [curr_inputs, prev_inputs]:
            if "pixel_values" in inputs:
                pv = inputs["pixel_values"]
                if pv.dim() == 3: 
                    inputs["pixel_values"] = pv.unsqueeze(0)
                elif pv.dim() == 5: 
                    inputs["pixel_values"] = pv[:, 0, ...]
                    
        # 3. [NEW] 分别进行两次 Forward
        try:
            curr_outputs = model(**curr_inputs)
            with torch.no_grad():
                prev_outputs = model(**prev_inputs)
        except RuntimeError as e:
            print(f"[Error] Batch {i} failed. Input shape: {curr_inputs.get('pixel_values', 'N/A').shape}")
            raise e
            
        # 4. 获取 Base Loss (通常 FIM 我们只基于当前步的 loss_val 作为基准，视你的需求而定)
        loss = None
        if isinstance(curr_outputs, dict) and "loss" in curr_outputs:
            loss = curr_outputs["loss"]
        elif hasattr(curr_outputs, "loss"):
            loss = curr_outputs.loss
        elif isinstance(curr_outputs, (list, tuple)):
            loss = curr_outputs[0]
            
        if loss is None:
            print("loss is none")
            continue
            
        base_loss_val = loss.item()
        
        # ==== [NEW] 计算时间差分 ====
        curr_pred = curr_outputs.output
        prev_pred = prev_outputs.output.detach() 
        
        # Ground truth 的整体差分 (如果你的 grad_loss 需要对比预测差分和真实差分)
        curr_labels = curr_batch["labels"].to(device)
        prev_labels = prev_batch["labels"].to(device)

        grad_loss = sobolev_loss_calc_denorm(
            dataset_name, 
            curr_pred,
            curr_labels, 
            order_control,                              
            base_loss_val, 
            loss_func, 
            device, 
            constants, 
            prev_pred=prev_pred,
            prev_labels=prev_labels,
            transpose=transpose, 
            balance_ratio=balance_ratio)
        # ==================================
        # 综合 loss
        loss = loss + grad_loss
        
        # 5. Backward
        model.zero_grad()
        loss.backward()
        
        if i == 0: 
            print(f"Current Batch Loss: {loss.item()}")
            for n, p in model.named_parameters():
                if "layers.0.blocks.0.attention.self.query" in n and p.grad is not None:
                    print(f"Gradient Norm for {n}: {p.grad.norm().item()}")
                    
        del curr_outputs, prev_outputs, loss
        torch.cuda.empty_cache() # 防止两次 forward 显存堆积

    # --- Post-process (Cholesky) ---
    Ldict = {}
    print("Computing Cholesky decomposition with Trace Normalization...")
    for name, cov in cov_dict.items():
        try:
            N = count_dict[name]
            Sigma = (cov / N).double()         
            dim = Sigma.shape[0]
            diag_mean = torch.diagonal(Sigma).mean().item()
            if care_I:
                if diag_mean > 0:
                    scale = 1.0 / diag_mean
                else:
                    scale = 1.0
                Sigma_final = (Sigma * scale) + torch.eye(dim, device=device, dtype=torch.float64)
                
            else:
                if diag_mean > 0:
                    jitter_val = max(diag_mean * 0.01, 1e-18) 
                else:
                    jitter_val = 1e-9
                
                Sigma_final = Sigma + torch.eye(dim, device=device, dtype=torch.float64) * jitter_val
            try:
                L_mat = torch.linalg.cholesky(Sigma_final)
            except RuntimeError:
                print(f"[Warning] Cholesky failed for {name}. Switching to Eigen...")
                eigenvalues, eigenvectors = torch.linalg.eigh(Sigma_final)
                min_eig = 1e-18 
                eigenvalues = torch.clamp(eigenvalues, min=min_eig)
                S_sqrt = torch.sqrt(eigenvalues)
                L_mat = eigenvectors * S_sqrt.unsqueeze(0)
            L_mat = L_mat.float()
            Lt = L_mat.t() 
            try:
                Lt_inv = torch.linalg.inv(Lt)
            except:
                Lt_inv = torch.linalg.pinv(Lt) 
            Ldict[name] = (Lt.cpu(), Lt_inv.cpu())
            del Sigma, Sigma_final, L_mat, Lt, Lt_inv
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"[Error] Processing {name}: {e}")
            dim = cov.shape[0]
            I_cpu = torch.eye(dim)
            Ldict[name] = (I_cpu, I_cpu)

    for h in handles: h.remove()
    model.eval()
    
    return Ldict