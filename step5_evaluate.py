import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    classification_report, confusion_matrix
)

# ── Load dữ liệu đã lưu từ step1 (không cần fetch lại) ──────────────
X_test = np.load('output/X_test.npy')
y_test = np.load('output/y_test.npy')
names  = np.load('output/names.npy', allow_pickle=True)

# ── Load model và features đã lưu ──────────────────────
best_model = joblib.load('output/svm_model.pkl')
scaler     = joblib.load('output/scaler.pkl')
X_test_feat   = np.load('output/features_test.npy')
X_test_scaled = scaler.transform(X_test_feat)

# ── Dự đoán ─────────────────────────────────────────────
y_pred = best_model.predict(X_test_scaled)

# ════════════════════════════════════════════════════════
#  CÁC CHỈ SỐ ĐÁNH GIÁ (theo bài giảng Phụ lục 1)
# ════════════════════════════════════════════════════════

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
rec  = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1   = f1_score(y_test, y_pred, average='macro', zero_division=0)

print("=" * 55)
print("  KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (Hold-out 75/25)")
print("=" * 55)
print(f"  Accuracy  (Độ đúng)     : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  Precision (Độ chính xác): {prec:.4f}  ({prec*100:.2f}%)")
print(f"  Recall    (Độ truy hồi) : {rec:.4f}  ({rec*100:.2f}%)")
print(f"  F1-Score                : {f1:.4f}  ({f1*100:.2f}%)")
print("=" * 55)

# ── Báo cáo chi tiết từng người ─────────────────────────
print("\nBÁO CÁO CHI TIẾT TỪNG LỚP:")
print(classification_report(y_test, y_pred, target_names=names))

# ════════════════════════════════════════════════════════
#  CONFUSION MATRIX + VISUALIZE
# ════════════════════════════════════════════════════════
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(14, 11))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=names, yticklabels=names)
plt.xlabel('Dự đoán (Predicted)', fontsize=12)
plt.ylabel('Thực tế (Actual)', fontsize=12)
plt.title(f'Confusion Matrix — Gabor Wavelets + SVM\n'
          f'Accuracy={acc*100:.1f}%  F1={f1*100:.1f}%',
          fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig('output/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("Đã lưu: output/confusion_matrix.png")

# Visualize kết quả dự đoán
fig, axes = plt.subplots(3, 6, figsize=(18, 9))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i], cmap='gray')
    pred_name = names[y_pred[i]]
    true_name = names[y_test[i]]
    correct   = y_pred[i] == y_test[i]
    ax.set_title(
        f"{'OK' if correct else 'SAI'}\nPred: {pred_name}\nTrue: {true_name}",
        color='green' if correct else 'red', fontsize=7
    )
    ax.axis('off')
plt.suptitle('Kết quả nhận dạng khuôn mặt — Gabor + SVM', fontsize=13)
plt.tight_layout()
plt.savefig('output/predictions.png', dpi=150, bbox_inches='tight')
plt.show()
print("Đã lưu: output/predictions.png")