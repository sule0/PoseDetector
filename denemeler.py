import cv2
import mediapipe as mp
import numpy as np

# Mediapipe Pose modelini başlatma
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Tişört resmi ve oranlarını tanımla
shirtpath = "1.png"
shirtRatio = 581 / 440
imgShirt = cv2.imread(shirtpath, cv2.IMREAD_UNCHANGED)  # Şeffaflık için UNCHANGED

# Video akışı başlatma
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    # Görüntüyü RGB'ye çevir
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Poz tespiti yap
    results = pose.process(img_rgb)

    # Tespit edilen noktaları çizdir
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Omuzların koordinatlarını al
        lmList = results.pose_landmarks.landmark
        left_shoulder = lmList[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = lmList[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        # Omuzlar arasındaki mesafeyi hesapla
        shoulder_width = int(((left_shoulder.x - right_shoulder.x) ** 2 +
                              (left_shoulder.y - right_shoulder.y) ** 2) ** 0.5 * img.shape[1]*1.5)

        # Tişörtün boyutlandırılması
        if shoulder_width > 0 and imgShirt is not None:
            resizedShirt = cv2.resize(imgShirt, (shoulder_width, int(shoulder_width * shirtRatio)))

            # Omuzların orta noktasını hesapla
            x_center = int((left_shoulder.x + right_shoulder.x) / 2 * img.shape[1])
            y_center = int((left_shoulder.y + right_shoulder.y) / 2 * img.shape[0])

            # Tişörtü omuzların biraz altına hizala
            y_offset = -40  # Omuzların biraz altına hizalamak için sabit bir offset (pixel cinsinden)
            x1 = int(x_center - resizedShirt.shape[1] / 2)
            y1 = int(y_center + y_offset)  # Tişörtün üst kısmını omuzların biraz altına hizala
            x2 = x1 + resizedShirt.shape[1]
            y2 = y1 + resizedShirt.shape[0]

            # Yükseklik ve genişlik uyumu
            x1, x2 = max(0, x1), min(img.shape[1], x2)
            y1, y2 = max(0, y1), min(img.shape[0], y2)

            # Tişörtü yerleştirme
            if y2 > y1 and x2 > x1:  # Geçerli bir bölge
                shirt_part = resizedShirt[:y2-y1, :x2-x1]
                alpha_s = shirt_part[:, :, 3] / 255.0  # Alfa kanalını normalize et
                alpha_l = 1.0 - alpha_s

                for c in range(0, 3):
                    img[y1:y2, x1:x2, c] = (alpha_s * shirt_part[:, :, c] +
                                            alpha_l * img[y1:y2, x1:x2, c])

    # Görüntüyü göster
    cv2.imshow("Image", img)

    # 'q' tuşuna basıldığında çıkış
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
