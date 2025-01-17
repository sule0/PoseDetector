import os
import cv2
from cvzone.PoseModule import PoseDetector
cap = cv2.VideoCapture(0)
detector = PoseDetector()
shirtpath = "1.png" 
ratio = 262 / 190  # Tişörtün en-boy oranını
shirtRatio = 581 / 440
imgShirt = cv2.imread(shirtpath, cv2.IMREAD_UNCHANGED)
rc = 0 #Bu değişkenler tişörtün boyutunu ve yerini hesaplamak için
lc = 0
ss = 10
while True:
    success, img = cap.read() # sonsuz döngü video akışı
    if not success:
        break 
    img = detector.findPose(img) #kişinin vucut hatlarını tespit ve görüntüye çizim
    lmList, bboxInfo = detector.findPosition(img, bboxWithHands=True, draw=True)
    if lmList:
        lm11 = lmList[21][1:4]  #omuz noktalarının koordinatları
        lm12 = lmList[22][1:4]
        widthOfShirt = int((lm11[0] - lm12[0]) * ratio) #omuzlar arası mesafe ve ratio'ya göre tişörtün genişliği
        if widthOfShirt > 0 and imgShirt is not None:  
            #tişörtü yeniden boyutlandırır
            resizedShirt = cv2.resize(imgShirt, (widthOfShirt, int(widthOfShirt * shirtRatio)))
            currentScale = (lm11[0] - lm12[0]) / 190 #tişörtün konumunu ve ölçeğini hesaplar
            offset = int(44 * currentScale), int(48 * currentScale)
            #tişörtü kişinin üzerine yerleştiren döngü, görüntüyü şeffaflık ile kişiye giydirir.
            try:
                for c in range(3):
                    img[lm12[1] - offset[1]:lm12[1] - offset[1] + resizedShirt.shape[0],
                        lm12[0] - offset[0]:lm12[0] - offset[0] + resizedShirt.shape[1], c] = \
                        img[lm12[1] - offset[1]:lm12[1] - offset[1] + resizedShirt.shape[0],
                            lm12[0] - offset[0]:lm12[0] - offset[0] + resizedShirt.shape[1], c] * \
                        (1 - resizedShirt[:, :, 3] / 255.0)
            except Exception as e:
                print(f"Error overlaying shirt: {e}")
    
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
cap.release()
cv2.destroyAllWindows()
