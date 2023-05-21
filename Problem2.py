import numpy as np
import cv2
import math
import glob

wpoints = np.zeros((9*6,3), np.float32)
wpoints[:,:2] = np.mgrid[21.5:193.5:9j,21.5:129:6j].T.reshape(-1,2)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
world_points = []
image_points = []
images = glob.glob("./Calibration_Imgs/*.jpg")
print("The Corner Points in each Image are displayed and saved onto the current directory")
for i,image in enumerate(images):
    current_img = cv2.imread(image)
    gray = cv2.cvtColor(current_img,cv2.COLOR_BGR2GRAY)
    #Finding corners of the checker board in each image given
    ret, corners = cv2.findChessboardCorners(gray, (9,6), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret == True:
        world_points.append(wpoints)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
        image_points.append(corners2)
        # Draw and display the corners
        current_img = cv2.drawChessboardCorners(current_img, (9,6), corners2, ret)
    cv2.namedWindow(f"Current_Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow(f"Current_Image", 1040, 780)
    cv2.imshow(f"Current_Image",current_img)
    cv2.imwrite(f"Corner_Image{i+1}.jpg",current_img)
    cv2.waitKey(0)
cv2.destroyAllWindows()

print("-----Finding Camera Calibration and Reprojection errors----------")
height,width = current_img.shape[:2]
ret, intrinsic_matrix, dist_coeff, rot_vects,trans_vects = cv2.calibrateCamera(world_points, image_points, gray.shape[::-1], None, None)

print("The Intrinsic Matrix K is:")
print(intrinsic_matrix)
#Calculating the reprojection errors using opencv project points function
mean_error = 0
for i in range(len(world_points)):
    image_points_calc, _ = cv2.projectPoints(world_points[i], rot_vects[i], trans_vects[i], intrinsic_matrix, dist_coeff)
    error = cv2.norm(image_points[i], image_points_calc, cv2.NORM_L2)/len(image_points_calc)
    print(f"The reprojection error for image {i+1} is ", error)
    mean_error += error
print( f"Mean Total Reprojection error is : {mean_error/len(world_points)}")