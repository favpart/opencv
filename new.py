import cv2 as cv ,cv2
import mss
import numpy as np
import time

with mss.mss() as sct:
    monitor = {"top": 0, "left": 0, "width": 800, "height": 640}
    temp_img = "img_tem/google.png"
    #template = cv.imread(temp_img,cv.IMREAD_GRAYSCALE)
    #template = cv.imread(temp_img,cv.IMREAD_ANYCOLOR)
    template = cv.imread(temp_img,cv.IMREAD_COLOR)
    
    while True:
        last_time = time.time()
        img = np.array(sct.grab(monitor))
        img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)
        #img = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)

        result = cv.matchTemplate(img,template,cv.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv.minMaxLoc(result)
        
        top_left = max_loc[0], max_loc[1]
        
        threshold = 0.7
        loc = np.where(result >= threshold)
        for pt in zip(*loc[::-1]):
            h, w = template.shape[:2]
            bottom_right = (pt[0] + w, pt[1] + h)
            cv.putText(img,f"{max_val:.2f}", pt,cv.FONT_ITALIC,0.6,(0,0,255),thickness=2)
            cv.rectangle(img, pt, bottom_right, color=(0,255,0),thickness=1)
        
        cv.putText(img,f"FPS {1 / (time.time() - last_time):.2f}", (5,20),cv.FONT_ITALIC,0.6,(0,255,0),thickness=2)

        winname = 'OpenCV'
        cv.imshow(winname, img)
        print(f"fps: {1 / (time.time() - last_time):.2f}")

        if cv2.waitKey(25) & 0xFF == 27 or cv2.getWindowProperty(winname,cv2.WND_PROP_VISIBLE) <1:
            cv2.destroyAllWindows()
            break