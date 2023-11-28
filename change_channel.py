import cv2
 
img_path = "./girl1.png"
save_path = './girl2.png'
  
  # 其实使用的方法非常简单，就是使用cv2.imread()读取四通道图片
  # 图片格式会自动转为三通道格式。
img = cv2.imread(img_path)
   
   # 再通过cv2.imwrite()直接保存，图片就保存为三通道
   # 之后用其他方式再读取就是三通道格式
cv2.imwrite(save_path, img)
