import cv2
import numpy as np


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
    loc = np.where(res >= threshold)  # 匹配程度大于%80的坐标y,x
    for pt in zip(*loc[::-1]):  # *号表示可选参数
        point = (pt[0], pt[1])
        point_list.append(point)
    return point_list


if __name__=="__main__":
    img_rgb = cv2.imread("C:/Users/admin/Desktop/image/source.JPG")
    template = cv2.imread('C:/Users/admin/Desktop/image/object.JPG')
    h, w = template.shape[:2]
    #调用opencv的模板匹配
    point_list=[]
    point_list = match_for_template(img_rgb, template, cv2.TM_CCOEFF_NORMED,0.55)

    # 模板匹配后符合要求的所有图案数量
    length = len(point_list)
    # 设定相同点的阈值
    equal_point_threshold = 4
    # 如果两个点距离在阈值范围内，则等同，然后用集合去重
    for i in range(length):
        for j in range(length):
            if abs(point_list[i][0] - point_list[j][0]) <= equal_point_threshold and abs(
                    point_list[i][1] - point_list[j][1]) <= equal_point_threshold:
                point_list[i] = point_list[j]
    point_set = set(point_list)

    images_list = []  # 记录截图信息，[坐标x,坐标y 图像array]
    for pt in point_set:  # *号表示可选参数
        right_bottom = (pt[0] + w, pt[1] + h)
        images_list.append((pt[0], pt[1], img_rgb[pt[1]:pt[1] + h, pt[0]:pt[0] + w]))

    #计算图像的RGB直方图，进行相关性比较。
    match_images = []#记录直方图
    template_hist = create_rgb_hist(template)
    for image in images_list:
        image_hist = create_rgb_hist(image[2])
        copare_result = cv2.compareHist(template_hist, image_hist, cv2.HISTCMP_CORREL)  # 计算两直方图的相关性
        if copare_result > 0.8:
            match_images.append(image)

    #输出匹配到的图片信息
    image_match_cord = []#记录每个匹配到的图片的坐标（top,left,right,bottom)
    for pt in match_images:
        right_bottom = (pt[0] + w, pt[1] + h)
        image_match_cord.append((((pt[0], pt[1]), (pt[0], pt[1] + h), (pt[0] + w, pt[1]), (pt[0] + w, pt[1] + h))))
        cv2.rectangle(img_rgb, (pt[0], pt[1]), right_bottom, (0, 0, 255), 1)
    cv2.imshow('Detected', img_rgb)
    cv2.waitKey(0)
    print(image_match_cord)

