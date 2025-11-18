# trainer/train_model.py - Huấn luyện với ảnh đã qua tiền xử lý
import cv2, os, numpy as np
from PIL import Image
from preprocessing import preprocess_face_pipeline

print("Đang huấn luyện mô hình LBPH với ảnh đã qua tiền xử lý...")

faces, ids = [], []

for file in os.listdir("dataset"):
    if file.endswith(".jpg") and file.startswith("User."):
        path = os.path.join("dataset", file)
        img = Image.open(path).convert("L")
        img_np = np.array(img, "uint8")
        
        # ÁP DỤNG CÙNG PIPELINE NHƯ LÚC CHỤP → đảm bảo nhất quán
        img_processed = preprocess_face_pipeline(img_np)
        
        faces.append(img_processed)
        id = int(file.split(".")[1])
        ids.append(id)

if len(faces) == 0:
    print("Không tìm thấy ảnh nào trong dataset!")
    exit()

os.makedirs("trainer", exist_ok=True)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(ids))
recognizer.write("trainer/trained_model.yml")

print(f"HOÀN TẤT! Đã train {len(set(ids))} người với {len(faces)} ảnh chất lượng cao")
print("→ Model đã lưu: trainer/trained_model.yml")