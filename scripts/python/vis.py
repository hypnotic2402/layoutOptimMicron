import cv2
import numpy as np
import time




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
    


