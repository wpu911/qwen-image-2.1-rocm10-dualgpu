# Qwen Image 2.1 在 ROCm10 异构双卡环境下的修复记录
**日期**：2026-09-28  
**主机**：AIGA / Ubuntu 24.04.3 LTS  
**主题**：Qwen Image 2.1 在 **R9700 + RX 7900 XTX** 双卡下的部署、崩溃修复与性能恢复

---

## 一、目标

在 ComfyUI 中部署并跑通 **Qwen Image 2.1**，实现如下双卡分工：

- **R9700 32GB**：负责 `Qwen Image 2.1 BF16 Diffusion 主模型`
- **RX 7900 XTX 24GB**：负责 `Qwen3-VL 8B Text Encoder` + `VAE`

最终目标：

1. 不再出现 `Reconnecting`
2. 不再出现 `Segmentation fault`
3. 双卡能够稳定生成图片
4. 性能恢复到正常水平

---

## 二、硬件与软件环境

### GPU

- **AMD Radeon AI PRO R9700 32GB**，架构 `gfx1201`
- **AMD Radeon RX 7900 XTX 24GB**，架构 `gfx1100`

### 软件

- **ROCm**：`10.0.0`
- **PyTorch**：`2.13.0+rocm10.0.0`
- **ComfyUI**：`0.37.4`

---

## 三、模型文件

- 主模型：`qwen_image_2.1_bf16.safetensors`
- Text Encoder：`qwen3vl_8b_int8_convrot.safetensors`
- VAE：`qwen_image_2.1_vae_bf16.safetensors`

组合：BF16 主模型 + INT8 Text Encoder + BF16 VAE。

---

## 四、最初问题

双卡工作流最开始存在两个问题：

1. **双卡第二张 GPU 一参与计算就 Segmentation fault**
2. 修复崩溃后虽然能出图，但 **2MP / 25 steps 需要 313~315 秒**，比单 R9700 慢约 7~8 倍

最小 PyTorch 测试显示：

```python
import torch
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
    print(torch.zeros(1, device=f"cuda:{i}"))
```

旧 runtime 下：

- 第一张 GPU 正常
- 第二张 GPU Segmentation fault
- 单独暴露任意一张卡时均正常

即典型的“谁排第二谁崩”。

---

## 五、第一类根因：ROCm10 KPACK 异构双卡 Bug

本机两张卡 ISA 不同：

```text
R9700      = gfx1201
7900 XTX   = gfx1100
```

旧版 HIP runtime 的 KPACK 逻辑只按第一张 GPU 的 ISA 选择 code object，再注册给所有可见设备。

异构 GPU 下会导致第二张卡拿到错误 ISA 的 code object，最终在 kernel launch 时崩溃。

上游修复逻辑改为：

- 按每种 GPU ISA 分组
- 为每种 ISA 单独加载对应 KPACK code object
- 再注册到对应设备

---

## 六、KPACK 修复结果

生产环境 `libamdhip64.so.7` 已换成包含异构 GPU KPACK 修复逻辑的版本。

修复后的 runtime 中可以看到：

```text
Could not load device type specific kpack object for ISA
Failed to load device type specific kpack object for ISA
Could not add device type specific kpack object
```

修复后：

- R9700 正常
- RX 7900 XTX 正常
- 两张卡同时可见时不再崩溃
- ComfyUI 不再因为第二 GPU 运算直接 `Reconnecting`

---

## 七、第二类问题：双卡性能坠落

KPACK 修好后，双卡已经能够完整生成，但日志中大量出现：

```text
HIPBLAS_STATUS_NOT_SUPPORTED
Will attempt to recover by calling cublas instead
```

说明 PyTorch 默认选择的 hipBLASLt 路径在当前 `gfx1201 + gfx1100` 异构组合下频繁失败，再 fallback 到其他 BLAS 实现。

这种“先撞墙再回退”的过程发生在大量 GEMM 上，最终导致性能严重下降。

### 修复前性能

```text
2MP / 25 steps
313 ~ 315 秒
约 12.4 秒 / step
```

单 R9700 参考约：

```text
41 秒总耗时
约 1.4 秒 / step
```

---

## 八、BLAS 性能修复

生产启动环境加入：

```bash
COMFY_AMD_FORCE_CUBLAS=1
```

并在 ComfyUI 启动阶段强制 PyTorch：

```python
torch.backends.cuda.preferred_blas_library("cublas")
```

在 ROCm 环境下，这会绕开当前异构双卡下异常的 hipBLASLt 路径，使用更稳定的 hipBLAS/rocBLAS 路线。

---

## 九、最终生产配置

### GPU 暴露项废

```bash
HIP_VISIBLE_DEVICES=1,0
CUDA_VISIBLE_DEVICES=1,0
COMFY_AMD_FORCE_CUBLAS=1
```

ComfyUI 内部对应：

```text
cuda:0 = R9700
cuda:1 = RX 7900 XTX
```

### 模型分工

```text
R9700 / cuda:0
└─ Qwen Image 2.1 BF16 Diffusion

RX 7900 XTX / cuda:1
├─ Qwen3-VL 8B INT8 Text Encoder
└─ Qwen Image 2.1 VAE BF16
```

---

## 十、正式工作流

### 文生图

`Qwen_Image_2.1_T2I_DualGPU_R9700_XTX.json`

### 图像编辑

`Qwen_Image_2.1_Edit_DualGPU_R9700_XTX.json`

### 设备绑定

```text
SelectModelDevice -> gpu:0
SelectCLIPDevice  -> gpu:1
SelectVAEDevice   -> gpu:1
```

此前 `widgets_values` 与 `widgets_values_named` 中存在 GPU 编号不一致的问题，已统一修正。

---

## 十一、最终实测

### KPACK 已修、BLAS 尚未修

```text
2MP / 25 steps
313 ~ 315 秒
```

### KPACK + BLAS 全部修复后

```text
2MP / 25 steps
42.05 秒
约 1.48 秒 / step
```

### 单 R9700 参考

```text
约 41 秒
```

结论：

> **双卡性能已恢复到正常水平。**

---

## 十二、双卡方案实际收益

这套双卡方案不是为了让 Diffusion 采样速度翻倍，因为主采样仍由 R9700 完成。

真正收益是把：

- Qwen3-VL 8B Text Encoder
- VAE

从 R9700 挪到 7900 XTX，使 R9700 将更多显存留给 Diffusion 主模型。

因此最终结构是：

```text
R9700
└─ Diffusion 主模型

RX 7900 XTX
├─ Text Encoder
└─ VAE
```

---

## 十三、相关备份

### ROCm10/KPACK 修改前

```text
/app/share/backup/comfyui-rocm10-prod-before-kpack-fix-20260928-183514
/app/share/backup/comfyui-rocm10-prod-before-kpack-fix-20260928-183527
```

### 强制 BLAS 修改前

```text
/app/share/backup/main.py.before-force-cublas-20260928-220600
```

### 双卡工作流修正前

```text
/app/share/backup/qwen21-dualgpu-workflows-before-prod-20260928-220943
```

---

## 十四、当前最终状态

截至 2026-09-28：

- [x] Qwen Image 2.1 模型正常
- [x] R9700 + 7900 XTX 异构双卡正常识别
- [x] 第二 GPU 不再 Segmentation fault
- [x] ComfyUI 双卡运行不再 `Reconnecting`
- [x] INT8 TE 正常运行于 XTX
- [x] BF16 VAE 正常运行于 XTX
- [x] BF16 Diffusion 正常运行于 R9700
- [x] 2MP / 25 steps 完整生成成功
- [x] 双卡性能恢复到约 42 秒
- [x] T2I 与 Edit 两套工作流均已修正

---

## 十五、以后升级 ROCm/PyTorch 时的最低验证

升级后首先执行：

```python
import torch

print(torch.__version__, torch.version.hip)
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
    print(torch.zeros(1, device=f"cuda:{i}"))
```

必须确认：

```text
cuda:0 PASS
cuda:1 PASS
```

然后再跑一次 2MP / 25 steps，并检查是否重新出现大量：

```text
HIPBLAS_STATUS_NOT_SUPPORTED
```

若大量出现并伴随性能下降，应首先检查 PyTorch BLAS backend 是否又回到了 hipBLASLt。

---

## 十六、一句话结论

> **Qwen Image 2.1 在 ROCm10 下的 R9700 + RX 7900 XTX 异构双卡最终修通：先修 KPACK 异构 GPU runtime 崩溃，再绕过异常 hipBLASLt 路径恢复性能，最终由 R9700 跑 BF16 Diffusion、XTX 跑 INT8 TE + BF16 VAE，稳定完成 2MP / 25 steps 出图。**
