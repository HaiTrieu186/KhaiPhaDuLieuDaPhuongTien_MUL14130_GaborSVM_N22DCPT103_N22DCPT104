import cv2, numpy as np

def extract_gabor_features(image, gabor_bank, n_blocks=2):
    """
    Trích xuất vector đặc trưng từ 1 ảnh.

    Ý tưởng cốt lõi(Block-based):
    - chia ảnh thành 4 vùng mỗi block 32x32 pixel (2 mắt, mũi, miệng),
        tính đặc trưng CỤC BỘ cho từng vùng rồi mới ghép lại.
          + block[0,0]=vùng trán/mắt trái,
          + block[0,1]=vùng mắt phải,
          + block[1,0]=vùng má/mũi trái,
          + block[1,1]=vùng má/cằm phải.

    Pipeline cho 1 ảnh:
      Ảnh 64×64
        ↓ chia thành 4 blocks (mỗi block 32×32 pixel)
      [Block_00][Block_01]
      [Block_10][Block_11]
        ↓ với mỗi block, áp 40 bộ lọc Gabor
      40 Response Maps / block
        ↓ tính Mean biên độ + Variance mỗi response map
      2 con số / (block × bộ lọc)
        ↓ ghép lại
      4 blocks × 40 bộ lọc × 2 thống kê = 320 chiều

    Tham số:
      image     : ảnh xám, giá trị float [0,1], shape (64, 64)
      gabor_bank: danh sách 40 bộ lọc từ build_gabor_bank()
      n_blocks  : số block theo mỗi chiều (mặc định 2 → lưới 2×2 = 4 vùng)

    Trả về:
      vector numpy shape (320,) — đặc trưng Block-based Gabor của ảnh
    """
    img = (image * 255).astype(np.uint8)
    h, w = img.shape          # 64 × 64

    bh = h // n_blocks        # chiều cao mỗi block = 32 pixel
    bw = w // n_blocks        # chiều rộng mỗi block = 32 pixel

    features = []

    # ── Duyệt qua từng block theo lưới n_blocks × n_blocks ──
    for i in range(n_blocks):         # hàng: 0 (trên) → 1 (dưới)
        for j in range(n_blocks):     # cột : 0 (trái) → 1 (phải)

            # Cắt vùng block từ ảnh gốc
            block = img[i*bh : (i+1)*bh,
                        j*bw : (j+1)*bw]   # shape: (32, 32)

            # Áp toàn bộ 40 bộ lọc Gabor lên block này
            for f in gabor_bank:
                # Tích chập cục bộ: response map của block (32×32)
                filtered = cv2.filter2D(block, cv2.CV_64F, f['kernel'])

                # Tại sao mean dùng abs?
                #   Response Gabor là sóng sin/cos → có thể âm.
                #   abs() đo cường độ cạnh (không phân biệt sáng→tối hay tối→sáng).
                #   Nếu không abs, mean ≈ 0 vì âm/dương triệt tiêu → mất thông tin.
                features.append(np.mean(np.abs(filtered)))   # Trung bình cục bộ

                # Variance tự bình phương nên không cần abs.
                # Đo mức độ phân tán của response trong block này.
                features.append(np.var(filtered))             # Phương sai cục bộ

    # Tổng số chiều: 4 blocks × 40 bộ lọc × 2 thống kê = 320
    return np.array(features)


# ════════════════════════════════════════════════════════════════
#  PHẦN CHẠY TRÍCH XUẤT ĐẶC TRƯNG CHO TOÀN BỘ DATASET
# ════════════════════════════════════════════════════════════════

from step2_gabor import build_gabor_bank
import time

X_train = np.load('output/X_train.npy')
X_test  = np.load('output/X_test.npy')

gabor_bank = build_gabor_bank()

# Thông báo cấu hình đang dùng
N_BLOCKS = 2
n_filters = len(gabor_bank)        # 40
n_features = N_BLOCKS * N_BLOCKS * n_filters * 2
print(f"Phương pháp: Block-based Gabor Features")
print(f"  Lưới blocks  : {N_BLOCKS}×{N_BLOCKS} = {N_BLOCKS**2} vùng")
print(f"  Số bộ lọc    : {n_filters} (5 tần số × 8 hướng)")
print(f"  Thống kê/lọc : 2 (Mean biên độ + Variance)")
print(f"  Vector output: {n_features} chiều")
print()

# ── Trích xuất tập train ─────────────────────────────
print(f"Trích xuất {X_train.shape[0]} ảnh train...")
t0 = time.time()
X_train_feat = np.array([
    extract_gabor_features(img, gabor_bank, n_blocks=N_BLOCKS)
    for img in X_train
])
print(f"  Xong! ({time.time()-t0:.0f}s)")

# ── Trích xuất tập test ──────────────────────────────
print(f"Trích xuất {X_test.shape[0]} ảnh test...")
t0 = time.time()
X_test_feat = np.array([
    extract_gabor_features(img, gabor_bank, n_blocks=N_BLOCKS)
    for img in X_test
])
print(f"  Xong! ({time.time()-t0:.0f}s)")

print(f"Shape features train: {X_train_feat.shape}")  # (300, 320)
print(f"Shape features test : {X_test_feat.shape}")   # (100, 320)

# ── Lưu lại ─────────────────────────────────────────
np.save('output/features_train.npy', X_train_feat)
np.save('output/features_test.npy',  X_test_feat)
print("Đã lưu! Chạy tiếp step4_train.py")