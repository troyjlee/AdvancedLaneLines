import numpy as np
import cv2
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

# global variables
bird_shape = (1500,7240)
num_windows = 1000
window_size = bird_shape[1]/num_windows
midpoint = 700

font = cv2.FONT_HERSHEY_SIMPLEX

# meters per pixel (12 feet = 724 pixels, 1 meter = 3.28 feet)
mpp = 12/(724*3.28)
y = np.arange(num_windows)*window_size + window_size/2

# import the needed variables
mtx = np.load('mtx.npy')
dist = np.load('dist.npy')
M = np.load('M.npy')
Minv = np.linalg.inv(M)
Minv = Minv/Minv[2,2]

# matrix to compute column sums in blocks
F = np.zeros(shape = (num_windows, bird_shape[1]))
for i in range(num_windows):
    F[i,int(i*window_size):int((i+1)*window_size)]= 1/window_size

def S_transformation(img):
    '''
    img should be an undistorted overhead view of the road.
    Normally the saturation channel is computed using two different 
    formulas, depending on the lightness of a pixel.  This function 
    only uses the saturation formula in the high lightness case.  This 
    prevents dark pixels have having high saturation.
    The function then adaptively thresholds the saturation value 
    and returns the result.
    '''
    scaled_img = img/255
    # we just take the formula for saturation in the high lightness case
    V_max = np.max(scaled_img,axis = 2)
    V_min = np.min(scaled_img,axis = 2)
    S = (V_max - V_min)/(2-(V_max + V_min)+1e-6)
    # rescale back into 0-255
    S = np.uint8(255*S/np.max(S))
    ret,thresh = cv2.threshold(S,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thresh = cv2.medianBlur(thresh,3)
    return thresh


def S_transformation2(img):
    '''
    img should be an undistorted normal view of the road.
    This function is similar to S_transformation but changes the order of operations.
    Here we
    1) Apply saturation formula in high lightness case
    2) Crop image to remove the sky
    3) Threshold
    4) Perspective transform to overhead view
    This order results in thinner detected lines than S_transformation which operates on 
    a perspective transformed image.
    '''
    scaled_img = img/255
    # we just take the formula for distortion in the bright L case
    V_max = np.max(scaled_img,axis = 2)
    V_min = np.min(scaled_img,axis = 2)
    S = (V_max - V_min)/(2-(V_max + V_min)+1e-6)
    S = np.uint8(255*S/np.max(S))
    S = S[400:,:]
    ret,thresh = cv2.threshold(S,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thresh = np.vstack((np.zeros(shape = (400,img.shape[1])),thresh)).astype(np.uint8)
    thresh = cv2.medianBlur(thresh,3)
    bird = cv2.warpPerspective(thresh,M,bird_shape)
    return bird

def L_transformation(img):
    '''
    img should be an undistorted overhead view of the road.
    This function is primarily for extracting a lane line when you have a good idea
    where it is and so can eliminate noise.
    We convert img to HLS and take the L channel.  We then threshold it with a hard 
    threshold of 200.
    '''
    hls = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    L = hls[:,:,1]
    ret,thresh = cv2.threshold(L,200,255,cv2.THRESH_BINARY)
    return thresh


def polyfit(proj):
    '''
    proj is a matrix.  This function finds the argmax along each row.  Zero rows are removed, 
    then a quadratic polynomial is fit according to the argmaxes (x values) and y values corresponding
    to the row numbers.
    '''
    indices = np.where(np.sum(proj,1) > 0)[0]
    maxes = np.argmax(proj[indices,:],1)
    y_nonzero = y[indices]
    p = np.polyfit(y_nonzero,maxes,2)
    return p


def process_image(img):
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    bird = cv2.warpPerspective(undistorted,M,bird_shape)
    # S_transformation2 seems to result in thinner lines and works 
    # better for the left lane
    #Sbird = S_transformation(bird)
    Sbird = S_transformation2(undistorted)
    #plt.imshow(Sbird,cmap='gray')
    #plt.show()
    proj = np.dot(F,Sbird)

    # pipeline for left lane
    left = proj[:,:midpoint]
    pleft = polyfit(left)
    # Radius of Curvature:  To convert to meters we multiply by mpp 
    # the conversion is simple as scale in x and y directions is the same
    left_roc = mpp * (1+(2*pleft[0]*bird_shape[1] + pleft[1])**2)**(3/2) / np.abs(2*pleft[0])

    # arrays of predicted points
    left_fitx = (pleft[0]*y**2+pleft[1]*y+pleft[2]).reshape((num_windows,1))
    left_pts = np.hstack((left_fitx,y.reshape((num_windows,1))))
    left_pts = left_pts.astype(np.int32)
    
    left_bottom = left_fitx[-1][0]

    # pipeline for right lane
    right = proj[:,midpoint:]
    candidate = np.argmax(np.average(right[-50:,:],0)) + midpoint 
    # if no start of line was found, we set candidate to be lane_width + location of left lane
    if np.abs(candidate - left_bottom - 724) > 50:
        candidate = left_bottom + 724

    # diff is the width of the lane 
    diff = candidate - left_bottom
    # we use the fit of the left lane to create a mask around where the right
    # lane should be
    guide = left_fitx + diff
    guide_points_l = np.hstack((guide - 250,y.reshape((num_windows,1))))
    guide_points_r = np.hstack((guide + 250,y.reshape((num_windows,1))))
    guide_draw = np.vstack((guide_points_l,np.flipud(guide_points_r))).astype(np.int32)
    warp_zero = np.zeros(shape = (7240,1500)).astype(np.uint8)
    cv2.fillPoly(warp_zero, [guide_draw], 255)
    L = L_transformation(bird)
    #plt.imshow(L,cmap = 'gray')
    #plt.show()
    masked_image = cv2.bitwise_and(L, warp_zero)
    proj_right = np.dot(F,masked_image)
    pright = polyfit(proj_right)
    #plt.imshow(masked_image, cmap = 'gray')	
    #plt.show()
        
    right_fitx = (pright[0]*y**2+pright[1]*y+pright[2]).reshape((num_windows,1))
    right_pts = np.hstack((right_fitx,y.reshape((num_windows,1))))

    right_pts = right_pts.astype(np.int32)
    right_roc = mpp * (1+(2*pright[0]*bird_shape[1] + pright[1])**2)**(3/2) / np.abs(2*pright[0])

    # for debugging
    #for i in range(len(proj_right)):
    #    cv2.circle(bird,(np.argmax(proj_right[i,:]),int(y[i])),15,(0,0,255),-1)
    #plt.imshow(bird[:,:,::-1])
    #plt.show()


    pts = np.vstack((left_pts, np.flipud(right_pts)))
    right_bottom = right_fitx[-1][0]
    
    # determine offset
    # x coordinate center of bird image is at 636
    offset = (2*636 - (left_bottom + right_bottom))*mpp

    # Create an image to draw the lines on
    warp_zero = np.zeros(shape = (7240,1500)).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    cv2.fillPoly(color_warp, [pts], (0,255,0))
    overlay = cv2.warpPerspective(color_warp,Minv,(img.shape[1],img.shape[0]))
    result = cv2.addWeighted(img, 1, overlay, 0.3, 0)
    roc = (left_roc + right_roc)/2
    roc_string = 'Radius of Curvature: {:0.2f}'.format(roc)
    offset_string = 'Lane Offset: {:0.2f} meters'.format(offset)
    cv2.putText(result,roc_string,(10,50), font, 1,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(result,offset_string,(10,100), font, 1,(255,255,255),2,cv2.LINE_AA)
    return result
    
def main():
    output_movie = 'project_annotated.mp4'
    clip1 = VideoFileClip('project_video.mp4')
    project_clip = clip1.fl_image(process_image)
    project_clip.write_videofile(output_movie, audio=False)
    
if __name__ == "__main__":
    main()
