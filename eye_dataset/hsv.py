import cv2
import numpy as np
import time
m,n,o,p,q,r=0,255,0,255,255,255
while True:
    # ret,frame=cap.read()
    kernal=np.ones((1,1),np.uint8)
    frame=cv2.imread(r"resume_stuff\eye_processing\eye data\1.jpg")
    # frame=cv2.imread(r"C:\Users\mail2\Desktop\swaminathan sir problems\eye data")
    frame=cv2.resize(frame ,(300,300))
    # cv2.resize(frame,(600,600))
    cv2.imshow("orignal",frame)
    img2=cv2.cvtColor(frame,cv2.COLOR_RGB2HSV)
    cv2.imshow("black white",img2)
    hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if m<=28 and n<=50:
        time.sleep(0.5)
        m+=1
        n=255
    elif m>28:
        m=0
        n=255
    l_b=np.array([m,n,o])
    u_b=np.array([p,q,r])
    # mask is removed from actual image it is directly taken from the track bars
    mask = cv2.inRange(hsv,l_b,u_b)
    # mask2=cv2.inRange(img2,m,n)
    closing=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernal,iterations=1)
    closing=cv2.morphologyEx(mask,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,kernal,iterations=1)
    # closing=cv2.morphologyEx(mask,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,kernal,iterations=1)
    res=cv2.bitwise_and(frame , frame, mask=mask)
    cv2.imshow("frame",frame)
    cv2.imshow("mask",mask)
    cv2.imshow("closed mask",closing)
    cv2.imshow("res",res)
    k=cv2.waitKey(1)
    if k==27:
        break
    print([m,n,o,p,q,r])
    # r+=8
    n-=2
# cap.release()
cv2.destroyAllWindows()
