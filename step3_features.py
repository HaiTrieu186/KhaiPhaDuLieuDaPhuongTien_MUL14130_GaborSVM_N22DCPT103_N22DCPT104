import cv2, numpy as np

def extract_gabor_features(image, gabor_bank):
    """
    Trích xuất vector 80 chiều từ 1 ảnh.
    Với mỗi trong 40 bộ lọc:
      1. Tích chập (convolution) ảnh với bộ lọc
      2. Tính Mean (trung bình biên độ)
      3. Tính Variance (phương sai)
    → 40 × 2 = 80 chiều
    """
    img      = (image * 255).astype(np.uint8)
    features = []

    for f in gabor_bank:
        filtered  = cv2.filter2D(img, cv2.CV_64F, f['kernel'])
        features.append(np.mean(np.abs(filtered)))  # Mean
        features.append(np.var(filtered))            # Variance

    return np.array(features)  # vector 80 chiều

# ── Load dữ liệu từ step1 ───────────────────────────
from step2_gabor import build_gabor_bank
import time

X_train = np.load('output/X_train.npy')
X_test  = np.load('output/X_test.npy')

gabor_bank = build_gabor_bank()

# ── Trích xuất (mất thời gian nhất) ─────────────────
print(f"Trích xuất {X_train.shape[0]} ảnh train...")
t0 = time.time()
X_train_feat = np.array([
    extract_gabor_features(img, gabor_bank) for img in X_train
])
print(f"  Xong! ({time.time()-t0:.0f}s)")

print(f"Trích xuất {X_test.shape[0]} ảnh test...")
t0 = time.time()
X_test_feat = np.array([
    extract_gabor_features(img, gabor_bank) for img in X_test
])
print(f"  Xong! ({time.time()-t0:.0f}s)")

print(f"Shape features train: {X_train_feat.shape}")  # (n,80)
print(f"Shape features test : {X_test_feat.shape}")

# ── Lưu lại (không phải chạy lại nữa) ──────────────
np.save('output/features_train.npy', X_train_feat)
np.save('output/features_test.npy',  X_test_feat)
print("Đã lưu! Chạy tiếp step4_train.py")