# capture_faces.py - CHỈ 1 Ô VUÔNG ĐẸP, GÓC NGHIÊNG VẪN CHỤP ĐƯỢC
import cv2, os, json, time, subprocess

os.makedirs("dataset", exist_ok=True)

# Load users
if os.path.exists("users.json"):
    with open("users.json", "r", encoding="utf-8") as f: users = json.load(f)
else: users = {}

next_id = str(max([int(k) for k in users.keys()], default=0) + 1)
name = input("Nhập tên người dùng: ").strip()
if not name: exit("Tên rỗng!")

users[next_id] = name
with open("users.json", "w", encoding="utf-8") as f:
    json.dump(users, f, ensure_ascii=False, indent=4)

# Load 2 cascade
c1 = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
c2 = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt2.xml")

cap = cv2.VideoCapture(0)
count = 0

print(f"\nChụp 25 ảnh cho: {name} (ID: {next_id})")

while count < 25:
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect bằng 2 cascade
    faces1 = c1.detectMultiScale(gray, 1.1, 4, minSize=(80,80))
    faces2 = c2.detectMultiScale(gray, 1.1, 3, minSize=(80,80))

    # Gộp + loại bỏ ô chồng (Non-Maximum Suppression đơn giản)
    all_faces = []
    for (x,y,w,h) in faces1: all_faces.append((x,y,w,h, w*h))
    for (x,y,w,h) in faces2: all_faces.append((x,y,w,h, w*h))

    if all_faces:
        # Lấy khuôn mặt lớn nhất (gần camera nhất)
        best = max(all_faces, key=lambda item: item[4])
        x,y,w,h = best[0], best[1], best[2], best[3]

        # Vẽ duy nhất 1 ô xanh
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.putText(frame, f"Chup: {count}/25 - SPACE", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.imshow("Chup anh", frame)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        if 'best' in locals():
            face = cv2.resize(gray[y:y+h, x:x+w], (200,200))
            cv2.imwrite(f"dataset/User.{next_id}.{count+1}.jpg", face)
            count += 1
            print(f"→ Đã chụp: {count}/25")
            time.sleep(0.15)
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if count >= 15:
    subprocess.call(["python", "trainer/train_model.py"])
    print("HOÀN TẤT")