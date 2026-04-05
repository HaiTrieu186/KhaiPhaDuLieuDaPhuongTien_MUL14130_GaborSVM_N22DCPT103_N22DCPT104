import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import joblib
import cv2
from PIL import Image, ImageTk
from step2_gabor import build_gabor_bank
from step3_features import extract_gabor_features

# ====================== LOAD DỮ LIỆU ======================
print("Đang load model và test set...")
model = joblib.load('output/svm_model.pkl')
scaler = joblib.load('output/scaler.pkl')
gabor_bank = build_gabor_bank()
X_test = np.load('output/X_test.npy')
y_test = np.load('output/y_test.npy')
names = np.load('output/names.npy', allow_pickle=True)

# Nhóm ảnh theo Person (từ test set)
from collections import defaultdict

groups = defaultdict(list)
for i, label in enumerate(y_test):
    groups[names[label]].append(i)

# Chuẩn bị danh sách Person
person_list = sorted(groups.keys())

# ====================== UI ======================
root = tk.Tk()
root.title("Demo Nhận Dạng Khuôn Mặt - Gabor + SVM")
root.geometry("1000x700")
root.configure(bg="#f0f0f0")

tk.Label(root, text="🎯 DEMO NHẬN DẠNG KHUÔN MẶT", font=("Arial", 18, "bold"), bg="#f0f0f0").pack(pady=10)

# Combobox chọn Person
tk.Label(root, text="Chọn người:", font=("Arial", 12), bg="#f0f0f0").pack()
combo_person = ttk.Combobox(root, values=person_list, state="readonly", font=("Arial", 11), width=30)
combo_person.pack(pady=5)

# Frame chứa 3 ảnh
frame_images = tk.Frame(root, bg="#f0f0f0")
frame_images.pack(pady=15)

photo_labels = []
buttons = []


def show_person_images(event=None):
    for widget in frame_images.winfo_children():
        widget.destroy()

    person = combo_person.get()
    if not person:
        return

    indices = groups[person][:3]  # tối đa 3 ảnh

    global photo_labels, buttons
    photo_labels = []
    buttons = []

    for order, idx in enumerate(indices, 1):
        # Tạo thumbnail
        img_array = (X_test[idx] * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_array).resize((200, 200))
        photo = ImageTk.PhotoImage(img_pil)

        # Label ảnh
        lbl = tk.Label(frame_images, image=photo, relief="solid", bd=2)
        lbl.image = photo  # giữ reference
        lbl.grid(row=0, column=order - 1, padx=10)
        photo_labels.append(lbl)

        # Button test
        btn = tk.Button(frame_images, text=f"Test ảnh {order}",
                        font=("Arial", 10), bg="#4CAF50", fg="white",
                        command=lambda i=idx, p=person: predict_image(i, p))
        btn.grid(row=1, column=order - 1, pady=5)
        buttons.append(btn)


combo_person.bind("<<ComboboxSelected>>", show_person_images)


# ====================== HÀM DỰ ĐOÁN ======================
def predict_image(img_idx, true_person):
    img_array = X_test[img_idx]

    # Trích đặc trưng
    features = extract_gabor_features(img_array, gabor_bank)
    features_scaled = scaler.transform([features])

    # Dự đoán
    pred_idx = model.predict(features_scaled)[0]
    pred_name = names[pred_idx]

    result = "✅ ĐÚNG" if pred_name == true_person else "❌ SAI"

    # Hiển thị popup kết quả
    messagebox.showinfo(
        "Kết quả dự đoán",
        f"{result}\n\n"
        f"Thực tế     : {true_person}\n"
        f"Model đoán : {pred_name}\n\n"
        f"Accuracy tổng thể: 91.00%"
    )

    # Hiện ảnh lớn
    display_img = (img_array * 255).astype(np.uint8)
    cv2.imshow(f"{result} - {true_person}", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ====================== CHẠY UI ======================
tk.Label(root, text="Chọn người → Chọn ảnh → Nhấn nút Test",
         font=("Arial", 10), fg="gray", bg="#f0f0f0").pack(pady=5)

root.mainloop()