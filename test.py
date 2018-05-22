import numpy as np
import cv2
import argparse

def l2_dist(line):
    x1,y1,x2,y2 = line[0]
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def on_same_line(line1, line2, threshold = 20):
    x1,y1,x2,y2 = line1
    x3,y3,x4,y4 = line2
    return np.abs(y3 - y1) < threshold

def choose_pair(pairs):
    max_crop = None
    max_mean = 0
    for pair in pairs:
        four_points = [pair[0][:2],pair[0][2:],pair[1][:2],pair[1][2:]]
        idx_order = np.argsort([x[0]+x[1] for x in four_points])
        top_left = four_points[idx_order[0]]
        below_right = four_points[idx_order[-1]]
        cropped = img[top_left[1]:below_right[1],top_left[0]:below_right[0]]
        if np.mean(cropped) > max_mean:
            max_mean = np.mean(cropped)
            max_crop = cropped.copy()
    return max_crop

def cropped_all(pairs):
    for idx, pair in enumerate(pairs):
        four_points = [pair[0][:2],pair[0][2:],pair[1][:2],pair[1][2:]]
        idx_order = np.argsort([x[0]+x[1] for x in four_points])
        top_left = four_points[idx_order[0]]
        below_right = four_points[idx_order[-1]]
        cropped = img[top_left[1]:below_right[1],top_left[0]:below_right[0]]
        print(idx)
        cv2.imwrite(args.output_path[:-4] + str(idx) + '.jpg', cropped)

parser = argparse.ArgumentParser(description='Crop the code box.')
parser.add_argument('-i', dest='input_path',default='images/cropped1.jpg', help='input path')
parser.add_argument('-o', dest='output_path', default='outputs/cropped.jpg', help='output_path')

args = parser.parse_args()


img = cv2.imread(args.input_path,0)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,10)
th3 = cv2.bitwise_not(th2)

# Horizontal Structure
kernel = np.ones((2,2),np.uint8)
horizontalsize = np.int(th3.shape[0]/5.)
horizontalstructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize, 1))
erode = cv2.erode(th3,horizontalstructure,iterations = 1)
dilate_h = cv2.dilate(erode, horizontalstructure, iterations = 1)
dilate_h = cv2.dilate(dilate_h,kernel,iterations = 2)



# Mix
# #structure_mix = cv2.bitwise_or(dilate_h,dilate_v)
# structure_mix = dilate_h.copy()
#
# # Image Dilation
# kernel = np.ones((3,3),np.uint8)
# bound_dilate = cv2.dilate(structure_mix, kernel, iterations=1)

# Find Bounding Box by the longest horizontal structure
# minLineLength = 100
# maxLineGap = 10
# lines = cv2.HoughLinesP(bound_dilate,1,np.pi/180,100,minLineLength,maxLineGap)
# image = img.copy()
# dist = [l2_dist(x) for x in lines]
# line_count = 0
# four_points = []
#
# for idx, line in enumerate(lines[np.flip(np.argsort(dist),0)]):
#     x1,y1,x2,y2 = line[0]
#     if idx != 0:
#         if np.abs(x1 - prev_x1) + np.abs(y1-prev_y1) < 10:
#             continue
#         line_count += 1
#     else:
#         line_count += 1
#     prev_x1, prev_y1 = x1 + 0, y1 + 0
#     four_points.extend([(x1,y1),(x2,y2)])
#     cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)
#     if line_count == 2:
#         break
#
# idx_order = np.argsort([x[0]+x[1] for x in four_points])
# top_left = four_points[idx_order[0]]
# below_right = four_points[idx_order[-1]]
# cv2.imwrite(args.output_path,img[top_left[1]:below_right[1],top_left[0]:below_right[0]])

minLineLength = 100
maxLineGap = 10
lines = cv2.HoughLinesP(dilate_h,1,np.pi/180,100,minLineLength,maxLineGap)
image = img.copy()
dist = [l2_dist(x) for x in lines]
line_pairs = []
for i in np.flip(np.argsort(dist),0):
    now_line = lines[i][0]
    for j in np.flip(np.argsort(dist),0):
        if i == j:
            continue
        else:
            test_line = lines[j][0]
            if not on_same_line(now_line, test_line) and \
            (np.abs(now_line[0]-test_line[0]) < 10 and np.abs(now_line[2]-test_line[2]) < 10):
                line_pairs.append((now_line, test_line))
                break

#cropped_all(line_pairs)
cropped_img = choose_pair(line_pairs)
cv2.imwrite(args.output_path,cropped_img)
