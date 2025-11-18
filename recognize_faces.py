# recognize_faces.py - CHỈ 1 Ô VUÔNG ĐẸP, GÓC NGHIÊNG CHUẨN
import cv2, json, numpy as np

c1 = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
c2 = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt2.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trained_model.yml")

with open("users.json", "r", encoding="utf-8") as f:
    users = json.load(f)

cap = cv2.VideoCapture(0)
print("RecoFace - Chỉ 1 ô vuông, nghiêng vẫn nhận diện ngon!")
print("Nhấn Q để thoát")

while True:
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    f1 = c1.detectMultiScale(gray, 1.1, 4, minSize=(80,80))
    f2 = c2.detectMultiScale(gray, 1.1, 3, minSize=(80,80))

    all_faces = []
    for (x,y,w,h) in f1: all_faces.append((x,y,w,h, w*h))
    for (x,y,w,h) in f2: all_faces.append((x,y,w,h, w*h))

    if all_faces:
        # Lấy khuôn mặt lớn nhất
        x,y,w,h = max(all_faces, key=lambda item: item[4])[:4]

        roi = cv2.resize(gray[y:y+h, x:x+w], (200,200))
        id_, conf = recognizer.predict(roi)

        if conf < 85:
            name = users.get(str(id_), "Unknown")
            text = f"{name} ({int(100-conf)}%)"
            color = (0,255,0)
        else:
            text = "Khong nhan dien"
            color = (0,0,255)

        # Vẽ 1 ô duy nhất
        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 3)
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.putText(frame, "Chi 1 o vuong - Nghieng van ngon!", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    cv2.imshow("RecoFace", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()