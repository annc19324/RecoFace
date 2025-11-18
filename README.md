git clone https://github.com/annc19324/RecoFace.git
cd RecoFace
pip install -r requirements.txt
python main.py

//xem các module trong file
D:\demo\RecoFace>findstr /r /n /i "^[ ]*import ^[ ]*from" *.py
capture_faces.py:2:import cv2, os, json, time, subprocess
delete_user.py:2:import os, json, shutil
main.py:2:import os, subprocess, sys
recognize_faces.py:2:import cv2, json, numpy as np
