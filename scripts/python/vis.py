import cv2
import numpy as np
import time


# out = cv2.VideoWriter('output_video.avi',0,1, (800,800))
# img1 = np.zeros((800, 800, 3), np.uint8) 
# cv2.rectangle(img1,(100,100) , (200,200) , (0,0,255) , 3)

# img2 = np.zeros((800, 800, 3), np.uint8) 
# cv2.rectangle(img2,(200,200) , (300,300) , (0,0,255) , 3)

# img3 = np.zeros((800, 800, 3), np.uint8) 
# cv2.rectangle(img3,(400,400) , (400,400) , (0,0,255) , 3)

# out.write(img1)
# out.write(img2)
# out.write(img3)

# cv2.destroyAllWindows()  
# out.release()


def genVid(XHis , wh):
    
    out = cv2.VideoWriter('output_video3.avi',0,40, (800,800))
    for frameX in XHis:
        img1 = np.zeros((800, 800, 3), np.uint8)
        X = frameX[1]
        for i in range(int(len(X) / 2)):
            # print(i)
            xi_min = X[2*i]
            yi_min = X[2*i + 1]
            xi_max = xi_min + wh[2*i]
            yi_max = yi_min + wh[2*i + 1]
            # print(xi_max)
            cv2.rectangle(img1, (int(xi_min) , int(yi_min)) , (int(xi_max) , int(yi_max)) , (0,0,255) , 3)

        out.write(img1)

    cv2.destroyAllWindows()  
    out.release()
    

        


# generate_video() 
# out.write(img)

# img = np.zeros((800, 800, 3), np.uint8) 
# cv2.rectangle(img,(200,200) , (300,300) , (0,0,255) , 3)
# out.write(img)
# cv2.imshow('image', img)
# time.sleep(3)
# cv2.imshow('image', img)
 
# Maintain output window until
# user presses a key
# cv2.waitKey(0)
 
# Destroying present windows on screen
# cv2.destroyAllWindows()

