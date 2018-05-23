import numpy as np
import os
import cv2
import argparse
import pytesseract

#os.removedirs('outputs')

parser = argparse.ArgumentParser(description='Crop the code box.')
parser.add_argument('-i', dest='input_path',default='images/cropped1.jpg', help='input path')
parser.add_argument('-o', dest='output_path', default='outputs/cropped.jpg', help='output_path')
parser.add_argument('--ocr', dest='ocr', help='Enable OCR detection', action='store_true')
parser.add_argument('--x-th', dest='x_th', type=int, default=30, help='X threshold, to consider both horizontal line can form a rectangle')
parser.add_argument('--y-th', dest='y_th', type=int, default=5, help='Y threshold, to consider both line are on the same line using diff-y')
parser.add_argument('--all', dest='save_all', help='Save all pair', action='store_true')
parser.set_defaults(ocr=False)
parser.set_defaults(save_all=False)

args = parser.parse_args()

def l2_dist(line):
    x1,y1,x2,y2 = line
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def on_same_line(line1, line2, threshold = args.y_th):
    x1,y1,x2,y2 = line1
    x3,y3,x4,y4 = line2
    not_on_left = max(x1,x2) > min(x3,x4)
    not_on_right = max(x3,x4) > min(x1,x2)
    return np.abs(y3 - y1) < threshold and not_on_left and not_on_right

def choose_pair(pairs):
    max_crop = None
    max_mean = 0
    for pair in pairs:
        four_points = [pair[0][:2],pair[0][2:],pair[1][:2],pair[1][2:]]
        idx_order = np.argsort([x[0]+x[1] for x in four_points])
        top_left = four_points[idx_order[0]]
        below_right = four_points[idx_order[-1]]
        print(four_points)
        cropped = img[top_left[1]:below_right[1],top_left[0]:below_right[0]]
        if args.ocr:
            text = pytesseract.image_to_string(cropped)
            if len(text) > 20 or len(text) < 16:
                continue
            print(text)
        if np.mean(cropped) > max_mean:
            max_mean = np.mean(cropped)
            max_crop = cropped.copy()
    cv2.imwrite(args.output_path,max_crop)

def cropped_all(pairs):
    for idx, pair in enumerate(pairs):
        four_points = [pair[0][:2],pair[0][2:],pair[1][:2],pair[1][2:]]
        idx_order = np.argsort([x[0]+x[1] for x in four_points])
        top_left = four_points[idx_order[0]]
        below_right = four_points[idx_order[-1]]
        cropped = img[top_left[1]:below_right[1],top_left[0]:below_right[0]]
        print(idx)
        cv2.imwrite(args.output_path[:-4] + str(idx) + '.jpg', cropped)

def merge_line(line1, line2):
    all_x = [line1[0],line1[2],line2[0],line2[2]]
    all_y = [line1[1],line1[3],line2[1],line2[3]]
    x1, x2 = np.min(all_x), np.max(all_x)
    y = np.int(np.mean(all_y))
    return [x1,y,x2,y]
    
def group_line(lines):
    lines_col = []
    line_num = len(lines)
    merged_index = np.zeros(line_num)
    for i in range(line_num):
        if merged_index[i] == 0:
            cur_line = lines[i][0]
            for j in range(line_num):
                if merged_index[j] != 0:
                    continue
                if on_same_line(cur_line, lines[j][0]):
                    #print(cur_line, lines[j][0])
                    cur_line = merge_line(cur_line, lines[j][0])
                    merged_index[j] = 1
            lines_col.append(cur_line)
    return lines_col


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
dilate_h = cv2.dilate(dilate_h,kernel,iterations = 1)



# Mix
# #structure_mix = cv2.bitwise_or(dilate_h,dilate_v)
# structure_mix = dilate_h.copy()
#
# # Image Dilation
# kernel = np.ones((3,3),np.uint8)
# bound_dilate = cv2.dilate(structure_mix, kernel, iterations=1)

minLineLength = 100
maxLineGap = 10
lines = cv2.HoughLinesP(dilate_h,0.1,np.pi/180,100,minLineLength,maxLineGap)
image = img.copy()
lines = group_line(lines)
dist = [l2_dist(x) for x in lines]
line_pairs = []
differ_threshold = args.x_th
for i in np.flip(np.argsort(dist),0):
    now_line = lines[i]
    for j in np.flip(np.argsort(dist),0):
        if i == j:
            continue
        else:
            test_line = lines[j]
            if not on_same_line(now_line, test_line) and np.abs(now_line[0]-test_line[0]) < differ_threshold\
             and np.abs(now_line[2]-test_line[2]) < differ_threshold:
                line_pairs.append((now_line, test_line))

# Chose
if args.save_all:
    cropped_all(line_pairs)
else:
    choose_pair(line_pairs)
