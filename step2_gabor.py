import cv2, numpy as np, matplotlib.pyplot as plt

def build_gabor_bank():
    """
    Xây dựng 40 bộ lọc Gabor (5 tần số - 8 hướng).
    Công thức: omega_n = (pi/2) * sqrt(2)^(-(n-1))
               theta_m = (pi/8) * m
               sigma   = pi / omega
    """
    filters      = []
    ksize        = 31   # Kích thước kernel (số lẻ)
    gamma        = 0.5  # Tỉ lệ khung hình: elip
    psi          = 0    # Pha

    for n in range(1, 6):           # 5 tần số
        omega = (np.pi/2) * (np.sqrt(2)**(-(n-1)))
        lambd = 2 * np.pi / omega
        sigma = np.pi / omega

        for m in range(8):          # 8 hướng
            theta = (np.pi/8) * m  # 0°,22.5°,...,157.5°

            kern = cv2.getGaborKernel(
                (ksize, ksize), sigma, theta, lambd, gamma, psi
            )
            kern /= (kern.sum() + 1e-8)  # Chuẩn hóa

            filters.append({
                'kernel'   : kern,
                'scale'    : n,
                'orient'   : m,
                'theta_deg': round(np.degrees(theta), 1)
            })

    return filters  # 40 bộ lọc

gabor_bank = build_gabor_bank()
print(f"Tổng bộ lọc: {len(gabor_bank)}")  # → 40

# ── Visualize 40 bộ lọc ──────────────────────────────
fig, axes = plt.subplots(5, 8, figsize=(16,10))
for i, (f, ax) in enumerate(zip(gabor_bank, axes.flat)):
    ax.imshow(f['kernel'], cmap='RdBu_r')
    ax.set_title(f"s{f['scale']} {f['theta_deg']}°", fontsize=7)
    ax.axis('off')
plt.suptitle('40 bộ lọc Gabor (5 tần số × 8 hướng)', fontsize=13)
plt.tight_layout()
plt.savefig('output/gabor_filters.png', dpi=100)
plt.show()
print("Xong! Chạy tiếp step3_features.py")