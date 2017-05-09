import numpy as np
import cv2

left_side = np.array([[290,661],[516,507],[538,491],[573,466],[580,461],[593,452],[597,449]],
    dtype = np.float32)
right_side = np.array([[1014,661],[774,507],[749,491],[710,466],[702,461],[687,452],[683,449]],
    dtype = np.float32)
src = np.vstack((left_side,right_side))

# pixels per foot
ppf = 724/12

left_dst = np.array([[290,120*ppf],[290,90*ppf],[290,80*ppf],[290,50*ppf],[290,40*ppf],
    [290,10*ppf],[290,0]],dtype = np.float32)
right_dst = np.array([[1014,120*ppf],[1014,90*ppf],[1014,80*ppf],[1014,50*ppf],[1014,40*ppf],
    [1014,10*ppf],[1014,0]],dtype = np.float32)
dst = np.vstack((left_dst,right_dst))

M = cv2.findHomography(src,dst)[0]
print(M)
np.save('M.npy',M)
