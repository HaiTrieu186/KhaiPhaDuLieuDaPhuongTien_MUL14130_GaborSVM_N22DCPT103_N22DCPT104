import tkinter as tk
from tkinter import ttk
import numpy as np
import joblib
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from step2_gabor import build_gabor_bank
from step3_features import extract_gabor_features

# ====================== LOAD DỮ LIỆU ======================
model = joblib.load('output/svm_model.pkl')
scaler = joblib.load('output/scaler.pkl')
gabor_bank = build_gabor_bank()
X_test = np.load('output/X_test.npy')
y_test = np.load('output/y_test.npy')
y_train = np.load('output/y_train.npy')
names = np.load('output/names.npy', allow_pickle=True)

from collections import defaultdict

groups = defaultdict(list)
for i, label in enumerate(y_test):
    groups[names[label]].append(i)

person_list = sorted(groups.keys())

# ====================== UI ======================
root = tk.Tk()
root.title("Demo Nhận Dạng Khuôn Mặt - Gabor Wavelets + SVM")
root.geometry("1180x720")
root.configure(bg="#f8f9fa")
root.resizable(False, False)

tk.Label(root, text="NHẬN DẠNG KHUÔN MẶT",
         font=("Arial", 22, "bold"), bg="#f8f9fa", fg="#1a73e8").pack(pady=(20, 5))
tk.Label(root, text="Local Gabor Wavelets (Block-based) + Support Vector Machine",
         font=("Arial", 11), bg="#f8f9fa", fg="#555555").pack(pady=(0, 10))

# Thanh thông tin Dataset
info_str = f" Dataset Olivetti: {len(names)} Đối tượng   |   📥 Tập Train: {len(y_train)} ảnh   |   📤 Tập Test: {len(y_test)} ảnh"
tk.Label(root, text=info_str, font=("Arial", 10, "bold"), bg="#e8f0fe", fg="#1a73e8", padx=20, pady=6,
         relief="flat").pack(pady=(0, 15))

main_frame = tk.Frame(root, bg="#f8f9fa")
main_frame.pack(padx=30, pady=5, fill="both", expand=True)

# Left panel
left = tk.LabelFrame(main_frame, text=" Chọn người và ảnh test ",
                     font=("Arial", 12, "bold"), bg="#f8f9fa", fg="#1a73e8", padx=15, pady=15)
left.pack(side="left", fill="y", padx=(0, 20))

tk.Label(left, text="Người:", font=("Arial", 11), bg="#f8f9fa").pack(anchor="w")
combo = ttk.Combobox(left, values=person_list, state="readonly", font=("Arial", 11), width=28)
combo.pack(pady=(8, 15), anchor="w")

thumb_frame = tk.Frame(left, bg="#f8f9fa")
thumb_frame.pack()

photo_refs = []
btn_refs = []


def update_thumbnails(*args):
    for w in thumb_frame.winfo_children():
        w.destroy()
    photo_refs.clear()
    btn_refs.clear()

    person = combo.get()
    if not person:
        return
    indices = groups[person][:3]

    for order, idx in enumerate(indices, 1):
        arr = (X_test[idx] * 255).astype(np.uint8)
        pil = Image.fromarray(arr).resize((170, 170))
        photo = ImageTk.PhotoImage(pil)

        lbl = tk.Label(thumb_frame, image=photo, relief="solid", bd=2, bg="white")
        lbl.image = photo
        lbl.grid(row=0, column=order - 1, padx=10)
        photo_refs.append(lbl)

        # Nút Test
        btn_test = tk.Button(thumb_frame, text=f"Test ảnh {order}", font=("Arial", 10, "bold"),
                             bg="#1a73e8", fg="white", width=14, height=2,
                             command=lambda i=idx, p=person: predict(i, p))
        btn_test.grid(row=1, column=order - 1, pady=5)

        # Nút Xem Gabor
        btn_gabor = tk.Button(thumb_frame, text="Xem Gabor", font=("Arial", 9),
                              bg="#4285f4", fg="white", width=14,
                              command=lambda i=idx: show_gabor_process(i))
        btn_gabor.grid(row=2, column=order - 1, pady=3)


combo.bind("<<ComboboxSelected>>", update_thumbnails)

# Right panel
right = tk.LabelFrame(main_frame, text=" Kết quả dự đoán ",
                      font=("Arial", 12, "bold"), bg="#ffffff", fg="#1a73e8", padx=20, pady=20)
right.pack(side="right", fill="both", expand=True)

result_text = tk.Label(right, text="Chưa có kết quả\n\nChọn người và nhấn nút Test ảnh",
                       font=("Arial", 12), bg="#ffffff", fg="#666666", justify="left")
result_text.pack(pady=10)

preview = tk.Label(right, bg="#f0f0f0", relief="solid", bd=3)
preview.pack(pady=15)


def reset():
    combo.set('')
    for w in thumb_frame.winfo_children():
        w.destroy()
    result_text.config(text="Chưa có kết quả\n\nChọn người và nhấn nút Test ảnh", fg="#666666")
    preview.config(image='')


tk.Button(root, text="Reset", font=("Arial", 10), bg="#f8f9fa", fg="#555555", command=reset).pack(side="bottom",
                                                                                                  pady=15)


# ====================== DỰ ĐOÁN ======================
def predict(img_idx, true_person):
    img_array = X_test[img_idx]

    # Block-based Gabor (320 chiều)
    features = extract_gabor_features(img_array, gabor_bank)
    features_scaled = scaler.transform([features])
    pred_idx = model.predict(features_scaled)[0]
    pred_name = names[pred_idx]

    is_correct = pred_name == true_person
    color = "#34a853" if is_correct else "#ea4335"
    status = "✅ ĐÚNG" if is_correct else "❌ SAI"

    result_text.config(
        text=f"Thực tế          : {true_person}\n"
             f"Mô hình dự đoán : {pred_name}\n\n"
             f"{status}",
        fg=color
    )

    display = (img_array * 255).astype(np.uint8)
    pil_big = Image.fromarray(display).resize((340, 340))
    photo_big = ImageTk.PhotoImage(pil_big)
    preview.config(image=photo_big)
    preview.image = photo_big


# ====================== MINH HỌA GABOR CỤC BỘ ======================
def show_gabor_process(img_idx):
    """
    Hiển thị minh họa Gabor cho phương pháp Block-based.
    Chỉ lấy ví dụ vùng Mắt Trái (Block 0,0) để hiển thị 8 Response Maps.
    """
    img = (X_test[img_idx] * 255).astype(np.uint8)
    h, w = img.shape

    bh, bw = h // 2, w // 2

    # Cắt lấy block [0, 0] (Vùng trán và mắt trái)
    block_00 = img[0:bh, 0:bw]

    # Lấy 8 bộ lọc tiêu biểu (scale 3 - tất cả hướng)
    selected = [f for f in gabor_bank if f['scale'] == 3]

    plt.figure(figsize=(14, 8))
    plt.suptitle("Minh họa Gabor Cục bộ - Phân tích vùng Mắt trái (Block 0,0) - Scale 3", fontsize=14,
                 fontweight='bold')

    # Ảnh gốc có vẽ khung đỏ bao quanh Block 0,0
    plt.subplot(3, 3, 1)
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.rectangle(img_color, (0, 0), (bw, bh), (255, 0, 0), 1)
    plt.imshow(img_color)
    plt.title("Ảnh gốc (Đỏ: Vùng trích xuất)", fontsize=11)
    plt.axis('off')

    # 8 response maps của riêng block_00
    for i, f in enumerate(selected):
        filtered = cv2.filter2D(block_00, cv2.CV_64F, f['kernel'])
        response = np.abs(filtered)

        plt.subplot(3, 3, i + 2)
        plt.imshow(response, cmap='jet')
        plt.title(f"Mắt Trái - Gabor {f['theta_deg']}°", fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.show(block=True)


root.mainloop()