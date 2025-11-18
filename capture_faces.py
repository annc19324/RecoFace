# capture_faces.py - Chụp ảnh + tiền xử lý ảnh trước khi lưu
import cv2, os, json, time, subprocess
from preprocessing import preprocess_face_pipeline

os.makedirs("dataset", exist_ok=True)

# Load hoặc tạo users.json
if os.path.exists("users.json"):
    with open("users.json", "r", encoding="utf-8") as f:
        users = json.load(f)
else:
    users = {}

next_id = str(max([int(k) for k in users.keys()], default=0) + 1)
name = input("Nhập tên người dùng: ").strip()
if not name:
    exit("Tên không được để trống!")

users[next_id] = name
with open("users.json", "w", encoding="utf-8") as f:
    json.dump(users, f, ensure_ascii=False, indent=4)

# Load 2 cascade để detect nghiêng tốt
c1 = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
c2 = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt2.xml")

cap = cv2.VideoCapture(0)
count = 0

print(f"\nĐang chụp 25 ảnh cho: {name} (ID: {next_id})")
print("→ Nhấn SPACE để chụp – Ảnh sẽ được xử lý tự động (giảm nhiễu, tăng sáng...)")

while count < 25:
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces1 = c1.detectMultiScale(gray, 1.1, 4, minSize=(80,80))
    faces2 = c2.detectMultiScale(gray, 1.1, 3, minSize=(80,80))
    all_faces = [(x,y,w,h,w*h) for x,y,w,h in faces1] + [(x,y,w,h,w*h) for x,y,w,h in faces2]

    if all_faces:
        x,y,w,h = max(all_faces, key=lambda item: item[4])[:4]
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.putText(frame, f"Chup: {count}/25 - Nhan SPACE", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.imshow("Chup anh - Dang xu ly anh tu dong", frame)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        if all_faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))
            
            # TIỀN XỬ LÝ ẢNH TRƯỚC KHI LƯU
            face_processed = preprocess_face_pipeline(face_roi)
            
            cv2.imwrite(f"dataset/User.{next_id}.{count+1}.jpg", face_processed)
            count += 1
            print(f"→ Đã chụp & xử lý ảnh thứ {count}/25")
            time.sleep(0.15)
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if count >= 15:
    print("\nĐang tự động huấn luyện lại mô hình...")
    subprocess.call(["python", "trainer/train_model.py"])
    print("HOÀN TẤT! Mô hình đã được cập nhật với ảnh chất lượng cao!")