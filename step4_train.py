import numpy as np, joblib, time
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# ── Load features đã trích xuất ─────────────────────
X_train_feat = np.load('output/features_train.npy')
X_test_feat  = np.load('output/features_test.npy')
y_train      = np.load('output/y_train.npy')

# ── Chuẩn hóa Z-score (công thức trong bài giảng) ───
# v' = (v - mean) / std
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_feat)
X_test_scaled  = scaler.transform(X_test_feat)

print(f"Sau chuẩn hóa - Mean: {X_train_scaled.mean():.4f}, Std: {X_train_scaled.std():.4f}")

# ── GridSearchCV tìm tham số tối ưu ─────────────────
# (5-fold cross-validation trên tập TRAIN)
param_grid = {
    'C'    : [0.1, 1, 10, 100],
    'gamma': ['scale', 0.001, 0.01, 0.1]
}

grid = GridSearchCV(
    SVC(kernel='rbf'),
    param_grid,
    cv=5,               # 5-fold CV
    scoring='f1_macro', # Tối ưu F1
    n_jobs=-1,
    verbose=1
)

print("Đang GridSearchCV (có thể mất 5-30 phút)...")
t0 = time.time()
grid.fit(X_train_scaled, y_train)
print(f"Xong! ({time.time()-t0:.0f}s)")
print(f"Tham số tốt nhất: {grid.best_params_}")
print(f"F1 trên CV      : {grid.best_score_:.4f}")

# ── Lưu model và scaler ──────────────────────────────
joblib.dump(grid.best_estimator_, 'output/svm_model.pkl')
joblib.dump(scaler,               'output/scaler.pkl')
print("Đã lưu: output/svm_model.pkl, output/scaler.pkl")
print("Chạy tiếp step5_evaluate.py")