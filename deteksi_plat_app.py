import streamlit as st
import cv2
import numpy as np
import imutils
import easyocr
from PIL import Image
import matplotlib.pyplot as plt

def process_image(image):
    # Konversi PIL Image ke array OpenCV
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Grayscale dan Blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
    edged = cv2.Canny(bfilter, 30, 200)  # Edge detection
    
    # Temukan kontur
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break
    
    if location is None:
        return img, None, "Plat nomor tidak ditemukan"
    
    # Buat mask dan crop plat nomor
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)
    
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]
    
    # Gunakan EasyOCR untuk membaca teks
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)
    
    # Gabungkan teks dari hasil EasyOCR
    text_elements_except_last = [item[-2] for item in result] if result else []
    text_to_display = ' '.join(text_elements_except_last) if text_elements_except_last else "Teks tidak terdeteksi"
    
    # Gambar persegi panjang pada plat nomor
    res = cv2.rectangle(img, tuple(location[0][0]), tuple(location[2][0]), (0, 255, 0), 3)
    
    return res, cropped_image, text_to_display

def main():
    st.title("Aplikasi Pendeteksi Plat Nomor")
    st.write("Unggah gambar kendaraan untuk mendeteksi plat nomor.")
    
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Baca gambar
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)
        
        # Proses gambar
        result_img, cropped_img, plate_text = process_image(image)
        
        if plate_text == "Plat nomor tidak ditemukan":
            st.warning("Plat nomor tidak ditemukan pada gambar.")
        else:
            st.success(f"Plat Terdeteksi: **{plate_text}**")
            
            # Tampilkan gambar dengan kotak hijau
            result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            st.image(result_img_rgb, caption="Gambar dengan plat nomor ditandai", use_column_width=True)
            
            if cropped_img is not None:
                st.image(cropped_img, caption="Area plat nomor yang di-crop", use_column_width=True)

if __name__ == "__main__":
    main()