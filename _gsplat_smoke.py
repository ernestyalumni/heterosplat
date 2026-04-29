import torch, gsplat

d = "cuda"
N = 10_000

means = torch.randn(N, 3, device=d)
quats = torch.randn(N, 4, device=d)
quats = quats / quats.norm(dim=-1, keepdim=True)
scales = torch.rand(N, 3, device=d) * 0.1
opacities = torch.rand(N, device=d)
colors = torch.rand(N, 3, device=d)
viewmats = torch.eye(4, device=d)[None]
Ks = torch.tensor([[[300., 0, 128.], [0, 300, 128.], [0, 0, 1.]]], device=d)

torch.cuda.synchronize()
img, alpha, meta = gsplat.rasterization(
    means, quats, scales, opacities, colors, viewmats, Ks, 256, 256
)
torch.cuda.synchronize()

print("shape:", tuple(img.shape), "device:", img.device)
print("min/max:", img.min().item(), img.max().item())
print("peak GPU mem MB:", torch.cuda.max_memory_allocated() / 1e6)
