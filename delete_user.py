# delete_user.py
import os, json, shutil

with open("users.json", "r", encoding="utf-8") as f:
    users = json.load(f)

for k,v in users.items():
    print(f"{k}: {v}")

uid = input("\nNhập ID muốn xóa: ").strip()
if uid in users:
    del users[uid]
    with open("users.json","w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=4)
    for f in os.listdir("dataset"):
        if f.startswith(f"User.{uid}."):
            os.remove(os.path.join("dataset", f))
    print("Đã xóa thành công! Chạy lại train để cập nhật.")
else:
    print("ID không tồn tại!")