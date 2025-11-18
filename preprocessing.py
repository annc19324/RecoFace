# preprocessing.py
# Pipeline xử lý ảnh số – CHUẨN GIÁO TRÌNH XỬ LÝ ẢNH SỐ (Gonzalez & Woods)
# Dùng cho cả: chụp ảnh → train → nhận diện realtime
import cv2
import numpy as np

def denoise_gaussian(img, ksize=5):
    """
    Bước 1: Giảm nhiễu bằng bộ lọc Gaussian
    - ksize: kích thước kernel (phải lẻ: 3,5,7,9...)
    - Mục đích: loại bỏ nhiễu hạt từ webcam rẻ hoặc ánh sáng yếu
    """
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def enhance_contrast_clahe(img, clip_limit=3.0, tile_grid_size=(8,8)):
    """
    Bước 2: Cân bằng histogram cục bộ (CLAHE)
    - Tăng độ tương phản ở những vùng tối/sáng không đều
    - Rất hiệu quả khi chụp trong phòng học, ánh sáng từ cửa sổ
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)

def adjust_gamma(img, gamma=1.2):
    """
    Bước 3: Gamma Correction – điều chỉnh độ sáng phi tuyến
    - gamma > 1.0: làm sáng ảnh bị tối
    - gamma < 1.0: làm tối ảnh bị chói
    - Giá trị 1.2 là tối ưu sau thử nghiệm thực tế
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def sharpen_unsharp_mask(img, sigma=1.0, strength=1.0):
    """
    Bước 4: Làm nét bằng Unsharp Mask
    - Tăng chi tiết vùng mắt, mũi, miệng → LBPH nhận diện tốt hơn
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def preprocess_face_pipeline(face_img):
    """
    PIPELINE HOÀN CHỈNH – ÁP DỤNG NHẤT QUÁN CHO TẤT CẢ GIAI ĐOẠN
    1. Giảm nhiễu Gaussian
    2. Tăng độ tương phản CLAHE
    3. Điều chỉnh độ sáng Gamma
    4. Làm nét Unsharp Mask
    """
    img = denoise_gaussian(face_img, ksize=5)
    img = enhance_contrast_clahe(img, clip_limit=3.0)
    img = adjust_gamma(img, gamma=1.2)
    img = sharpen_unsharp_mask(img, sigma=1.0, strength=1.0)
    return img