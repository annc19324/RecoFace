# main.py
import os, subprocess, sys

def run(cmd):
    subprocess.call(cmd, shell=True)

while True:
    os.system("cls" if os.name == "nt" else "clear")
    print("="*50)
    print("     RECOFACE - NHẬN DIỆN KHUÔN MẶT")
    print("="*50)
    print("1. Thêm người mới (chụp ảnh)")
    print("2. Huấn luyện lại mô hình")
    print("3. Chạy nhận diện realtime")
    print("4. xóa người dùng")
    print("5. Thoát")
    print("-"*50)
    choice = input("Chọn (1-5): ").strip()

    if choice == "1":
        run("python capture_faces.py")
    elif choice == "2":
        run("python trainer/train_model.py")
    elif choice == "3":
        run("python recognize_faces.py")
    elif choice == "4": 
        run(["python", "delete_user.py"])
    elif choice == "5":
        print("Tạm biệt!")
        break
    else:
        print("Chỉ chọn 1-5 thôi!")
        os.system("pause" if os.name == "nt" else "read -p 'Enter...'")