from cv2.cv import *

img = LoadImage("NRF24L01-1.jpg")
NamedWindow("opencv")
ShowImage("opencv",img)
WaitKey(0)
