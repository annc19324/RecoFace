# trainer/train_model.py
import cv2, os, numpy as np
from PIL import Image

print("Đang huấn luyện mô hình...")

faces, ids = [], []
for file in os.listdir("dataset"):
    if file.endswith(".jpg") and file.startswith("User."):
        path = os.path.join("dataset", file)
        img = Image.open(path).convert("L")
        img_np = np.array(img, "uint8")
        id = int(file.split(".")[1])
        faces.append(img_np)
        ids.append(id)

if len(faces) == 0:
    print("Không tìm thấy ảnh nào trong dataset!")
    exit()

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(ids))
recognizer.write("trainer/trained_model.yml")

print(f"HOÀN TẤT! Đã train {len(set(ids))} người với {len(faces)} ảnh")
print("→ Model đã lưu: trainer/trained_model.yml")