import cv2
from cvzone.PoseModule import PoseDetector

def process_image(image_path, shirt_path):
    # Resimleri yükleme
    img = cv2.imread(image_path)
    imgShirt = cv2.imread(shirt_path, cv2.IMREAD_UNCHANGED)
    
    if imgShirt is None:
        print("T-shirt image could not be loaded. Please check the file path.")
        return

    # Pose dedektörü oluşturma
    detector = PoseDetector()
    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img, bboxWithHands=True, draw=False)

    if lmList:
        # Omuz noktalarıq
        lmLeftShoulder = lmList[11][1:4]
        lmRightShoulder = lmList[12][1:4]
        lmMidHip = lmList[23][1:4]  # Karın bölgesinin Y koordinatını almak için

        # Tişörtün yerleştirileceği pozisyonu hesaplama
        x_offset = int((lmLeftShoulder[0] + lmRightShoulder[0]) / 2 - imgShirt.shape[1] / 2)
        y_offset = int(lmMidHip[1] + imgShirt.shape[0] / 3)  # Karın bölgesi

        # Tişörtün yerleştirileceği pozisyonun sınırlarını kontrol etme
        x_end = min(x_offset + imgShirt.shape[1], img.shape[1])
        y_end = min(y_offset + imgShirt.shape[0], img.shape[0])

        x_start = max(x_offset, 0)
        y_start = max(y_offset, 0)

        shirt_crop_x = x_start - x_offset
        shirt_crop_y = y_start - y_offset

        croppedShirt = imgShirt[shirt_crop_y:shirt_crop_y + (y_end - y_start), shirt_crop_x:shirt_crop_x + (x_end - x_start)]

        # Tişörtü yerleştirme
        for c in range(3):
            try:
                img[y_start:y_end, x_start:x_end, c] = \
                    croppedShirt[:, :, c] * (croppedShirt[:, :, 3] / 255.0) + \
                    img[y_start:y_end, x_start:x_end, c] * \
                    (1 - croppedShirt[:, :, 3] / 255.0)
            except ValueError as e:
                print(f"Error overlaying shirt: {e}")
                return

    # Sonuç görüntüsünü göster
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Kullanım
process_image("person.jpg", "1.png")