from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import numpy as np, matplotlib.pyplot as plt, os

os.makedirs('output', exist_ok=True)

# ── Tải LFW dataset ──────────────────────────────────
print("Đang tải dataset LFW...")
lfw = fetch_lfw_people(
    min_faces_per_person=50,  # Người có >= 50 ảnh
    resize=0.5,               # Ảnh 62x47 pixel
    color=False               # Ảnh xám
)

X     = lfw.images        # (n_samples, 62, 47)
y     = lfw.target        # (n_samples,)
names = lfw.target_names  # Tên từng người

print(f"Số người  : {len(names)}")
print(f"Tổng ảnh  : {X.shape[0]}")
print(f"Kích thước: {X.shape[1]} x {X.shape[2]}")
print()
for i, name in enumerate(names):
    print(f"  {name}: {(y==i).sum()} ảnh")

# ── Hold-out 75/25 + Stratified sampling ─────────────
# (Chia dataset thành 2 phần Train - Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,    # 25% test, 75% train
    random_state=42,
    stratify=y         # Stratified: tỉ lệ lớp đều nhau
)

print(f"Train : {X_train.shape[0]} ảnh (75%)")
print(f"Test  : {X_test.shape[0]} ảnh (25%)")

# ── Lưu để các file sau dùng lại ─────────────────────
np.save('output/X_train.npy', X_train)
np.save('output/X_test.npy',  X_test)
np.save('output/y_train.npy', y_train)
np.save('output/y_test.npy',  y_test)
np.save('output/names.npy',   names)
print("Đã lưu: output/X_train.npy, X_test.npy, ...")

# ── Hiển thị mẫu ảnh ─────────────────────────────────
fig, axes = plt.subplots(2, 6, figsize=(15,5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i], cmap='gray')
    ax.set_title(names[y_train[i]].split()[-1], fontsize=8)
    ax.axis('off')
plt.suptitle('Mẫu ảnh dataset LFW', fontsize=12)
plt.tight_layout()
plt.savefig('output/sample_images.png', dpi=100)
plt.show()
print("Xong! Chạy tiếp step2_gabor.py")