# QRcode-location 这是一个二维码定位的实例
This is an Example about how to locate QRcode.
## 克隆后请先对opencv环境进行配置
Please configure the opencv environment after cloning
## 实现的内容
该程序可以实现图片的二维码定位，其主要运用的原理是通过二值化和opencv自带的轮廓筛选功能在图片中找到二维码的位置，然后利用向量知识通过三角形将二维码的位置进行矫正。  
The program can locate the QRcode in a picture. Its main principle is using binarization and the opencv-contour-selection function to find the position of QRcode in the picture, and then rectify the position of the qrcode by using vector knowledge through triangles.
## 处理过程
QRcode_before
![QRcode_before](/img/QRcode_before.jpg) 
threshold
![threshold](/img/threshold.jpg) 
canvas
![canvas](/img/canvas.jpg)
QRcode
![QRcode](/img/QRcode.jpg)  
