#图像中的目标定位

##1、传统方法

###1.1 使用OpenCV 模板匹配 +基于颜色直方图匹配
**使用说明**

- 需要安装opencv,numpy库
- 在“传统方法”目录下，使用jupyter nodebook 打开“传统方法的目标定位.ipynb”
- 或者打开“tradition_method.py”修改目标路径和源图片的路径即可执行

		程序42、43行
		img_rgb = cv2.imread("C:/Users/admin/Desktop/image/source.JPG")
    	template = cv2.imread('C:/Users/admin/Desktop/image/object.JPG')

**原理:**

- 使用模板匹配在图像中寻找物体
- OpenCV函数：cv2.matchTemplate()
- 对输入的两张图像进行直方图均衡化及直方图计算步骤后，可以对两个图像的直方图进行对比

**过程**

1. 模板匹配就是用来在大图中找小图，也就是说在一副图像中寻找另外一张模板图像的位置。
	目标物体：
![](https://i.imgur.com/A3HMdAS.jpg)
	
	源图像：
	![](https://i.imgur.com/KVmdVmi.jpg)
2. 用cv2.matchTemplate()实现模板匹配。首先我们来读入图片和模板，设定一个匹配阈值（阈值较低）来**归一化相关匹配**多个物体。由于可能匹配到其他物体，因此需要进一步处理。

		def match_for_template(image2match,template,method,threshold):
		'''
    	基于opencv的模板匹配，使用模板匹配在图像中寻找物体
    	:param image2match:需要匹配的图像
    	:param template:需要寻找的物体图像
    	:param method:这里选用归一化相关匹配
    	:param threshold:
		:return:
    	'''
    	res = cv2.matchTemplate(image2match, template, method)
    	point_list = []
    	loc = np.where(res >= threshold)  # 匹配程度大于threshold的坐标y,x
    	for pt in zip(*loc[::-1]):  # *号表示可选参数
        	point = (pt[0], pt[1])
        	point_list.append(point)
    	return point_list

3. 根据返回的所有物体进行去重。
4. 考虑到目标物体与其他物体颜色上区别很大，因此可以采用基于颜色的特征的直方图进行比较。这里采用了一种基于区间统计的颜色直方图方法，它是将R,G,B各分量的颜色信息划分为16 区间。比如；将图像每个像素的R、G和B的值都分成16个bins(256除以16)，构建RGB（16X16X16 ）颜色模型，然后再统计各个区间内的像素数。

		def create_rgb_hist(image):
    	'''
    	创建R,G,B直方图
    	:param image: 输入图片
    	:return:np.array类型的直方图
    	'''
    	h, w, c = image.shape
		rgHist = np.zeros([16 * 16 * 16, 1], np.float32)  # 必须是float型
    	hsize = 256 / 16
    	for row in range(0, h, 1):
        	for col in range(0, w, 1):
            	b = image[row, col, 0]
            	g = image[row, col, 1]
            	r = image[row, col, 2]
            	index = np.int(b / hsize) * 16 * 16 + np.int(g / hsize) * 16 + np.int(r / hsize)
            	rgHist[np.int(index), 0] = rgHist[np.int(index), 0] + 1
    	return rgHist
	
直方图比较：采用待识别图像与目标图像的直方图**相关性系数**进行计算得出结果。

		match_images = []
		template_hist = create_rgb_hist(template)
		for image in images_list:
    		image_hist = create_rgb_hist(image[2])
    		copare_result = cv2.compareHist(template_hist, image_hist, cv2.HISTCMP_CORREL)  # 计算两直方图的相关性
    		if copare_result > 0.8:
        		match_images.append(image)

![](https://i.imgur.com/zriE39v.jpg)

[((359, 143), (359, 162), (395, 143), (395, 162)), ((142, 121), (142, 140), (178, 121), (178, 140)), ((567, 163), (567, 182), (603, 163), (603, 182)), ((271, 123), (271, 142), (307, 123), (307, 142)), ((184, 123), (184, 142), (220, 123), (220, 142)), ((144, 141), (144, 160), (180, 141), (180, 160)), ((317, 141), (317, 160), (353, 141), (353, 160)), ((444, 123), (444, 142), (480, 123), (480, 142)), ((357, 123), (357, 142), (393, 123), (393, 142)), ((272, 143), (272, 162), (308, 143), (308, 162)), ((142, 162), (142, 181), (178, 162), (178, 181)), ((186, 143), (186, 162), (222, 143), (222, 162)), ((444, 164), (444, 183), (480, 164), (480, 183)), ((97, 164), (97, 183), (133, 164), (133, 183)), ((184, 164), (184, 183), (220, 164), (220, 183)), ((99, 143), (99, 162), (135, 143), (135, 162)), ((399, 121), (399, 140), (435, 121), (435, 140)), ((397, 142), (397, 161), (433, 142), (433, 161)), ((445, 143), (445, 162), (481, 143), (481, 162)), ((316, 121), (316, 140), (352, 121), (352, 140))]

**优点**

- 使用opencv的工具库
- 原理相对简单
- 实现容易

###1.2 其他方法：HOG特征提取算法：
**HOG特征提取算法**

- 读入所需要的检测目标即输入的image
- 将图像进行灰度化（将输入的彩色的图像的r,g,b值通过特定公式转换为灰度值）
- 采用Gamma校正法对输入图像进行颜色空间的标准化（归一化）
- 计算图像每个像素的梯度（包括大小和方向），捕获轮廓信息
- 统计每个cell的梯度直方图（不同梯度的个数），形成每个cell的descriptor
- 将每几个cell组成一个block（以3*3为例），一个block内所有cell的特征串联起来得到该block的HOG特征descriptor
- 将图像image内所有block的HOG特征descriptor串联起来得到该image（检测目标）的HOG特征descriptor，这就是最终分类的特征向量



##2、机器学习/深度学习方法

###2.1 SVM 算法

**使用说明**

- 需要安装opencv,numpy，sklearn库
- 在“机器学习方法”目录下，使用jupyter nodebook 打开“SVM目标定位.ipynb”

**原理:**

- svm 支持向量机 (Support Vector Machines)，请参考原理

**过程**


1. 训练集、测试集。截取部分图片，进行滑窗截图。并将RGB图像进行转换为列向量
		
		#遍历训练集，生成训练集数据。
		img_list=[]#训练集列表
		train_y=[]#训练集标签
		for filename in os.listdir("./image/train/0"):
    		file="./image/train/0/"+filename
    		img_rgb=cv2.imread(file)
   	 		img_array=list(img_rgb.reshape(-1))
    		img_list.append(img_array)
    		train_y.append(0)
		for filename in os.listdir("./image/train/1"):
    		file="./image/train/1/"+filename
    		img_rgb=cv2.imread(file)
    		img_array=list(img_rgb.reshape(-1))
    		img_list.append(img_array)
    		train_y.append(1)


2. 归一化预处理，使用sklearn.preprocessing.StandardScaler类，使用该类的好处在于可以保存训练集中的参数（均值、方差）直接使用其对象转换测试集数据。

3. 使用“rgb"核函数进行拟合。在拟合后, 这个模型可以用来预测新的值。

**优点**

- 在高维空间中非常高效.
- 即使在数据维度比样本数量大的情况下仍然有效.
- 本例中，训练集的特征数为2000，训练集为1000以下，故采用SVM算法进行回归学习 

**结果，待优化**
              
				precision    recall  f1-score   support

          			 0       1.00      0.87      0.93       139
          			 1       0.22      1.00      0.36         5




###2.2 其他算法

- over feat
- R-CNN
- Fast R-CNN
- YOLO