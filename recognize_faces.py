# recognize_faces.py - Nhận diện realtime với tiền xử lý ảnh
import cv2, json, numpy as np, os
from preprocessing import preprocess_face_pipeline

c1 = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
c2 = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt2.xml")

if not os.path.exists("trainer/trained_model.yml"):
    print("Chưa có model! Hãy thêm người và train trước.")
    input("Nhấn Enter để thoát...")
    exit()

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trained_model.yml")

with open("users.json", "r", encoding="utf-8") as f:
    users = json.load(f)

cap = cv2.VideoCapture(0)
print("RecoFace đang chạy - Nhấn Q để thoát")

while True:
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    f1 = c1.detectMultiScale(gray, 1.1, 4, minSize=(80,80))
    f2 = c2.detectMultiScale(gray, 1.1, 3, minSize=(80,80))
    all_faces = [(x,y,w,h,w*h) for x,y,w,h in f1] + [(x,y,w,h,w*h) for x,y,w,h in f2]

    if all_faces:
        x,y,w,h = max(all_faces, key=lambda item: item[4])[:4]
        
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (200, 200))
        
        # TIỀN XỬ LÝ ẢNH TRƯỚC KHI DỰ ĐOÁN
        face_processed = preprocess_face_pipeline(face_roi)
        
        id_, conf = recognizer.predict(face_processed)

        if conf < 85:
            name = users.get(str(id_), "Unknown")
            text = f"{name} ({int(100-conf)}%)"
            color = (0,255,0)
        else:
            text = "Khong nhan dien"
            color = (0,0,255)

        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 3)
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.putText(frame, "Nhan Q de thoat", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    cv2.imshow("RecoFace - Xu ly anh truoc khi nhan dien", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()