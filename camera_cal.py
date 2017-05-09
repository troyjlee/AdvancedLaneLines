import numpy as np
import cv2


def create_grid(num_x,num_y,size_x = 1., size_y = 1.):
    '''
    create_grid returns a matrix whose columns are (x,y,z) coordinates 
    of a num_x-by-num_y grid.
    all z values are set to 0
    the points are given going row by row, where the x step size is size_x 
    and the y step size is size_y
    '''
    x = np.linspace(0,(num_x-1)*size_x, num_x)
    y = np.linspace(0,(num_y-1)*size_y, num_y)
    x_grid = np.tile(x,num_y).reshape(num_x*num_y,1)
    y_grid = np.dot(np.diag(y),np.ones((num_y,num_x)))
    y_grid = y_grid.reshape((num_x*num_y,1))
    return np.hstack((x_grid,y_grid,np.zeros((num_x*num_y,1)))).astype(np.float32)




def create_obj_and_img_points(x_step = 1.0, y_step =1.0):
    '''
    this function compiles arrays of objpoints and imgpoints by iterating over 
    all calibration images.
    we try to detect either (9,5) or (9,6) grid intersections with 
    findChessboardCorners.
    the objpoints points are created by calling create_grid with the 
    appropriate grid size
    '''
    objpoints = []
    imgpoints = []
    for i in range(1,21):
        file_path = './camera_cal/calibration'+str(i)+'.jpg'
        img = cv2.imread(file_path)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # try (9,6) and (9,5)
        ret96, corners96 = cv2.findChessboardCorners(img, (9,6), None)
        ret95, corners95 = cv2.findChessboardCorners(img, (9,5), None)
        if ret96:
            print('True 96 image {}'.format(i))
            objpoints.append(create_grid(9,6,size_x = x_step, size_y = y_step))
            imgpoints.append(corners96)
        elif ret95:
            print('True 95 image {}'.format(i))
            objpoints.append(create_grid(9,5,size_x = x_step, size_y = y_step))
            imgpoints.append(corners95)
    return objpoints, imgpoints



def main():
    file_path = './camera_cal/calibration'+str(1)+'.jpg'
    img = cv2.imread(file_path)
    objpoints, imgpoints = create_obj_and_img_points()
    ret,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(objpoints,imgpoints, img.shape[:2],
        None, None)
    np.save('mtx.npy',mtx)
    np.save('dist.npy',dist)

if __name__ == "__main__":
    main()
