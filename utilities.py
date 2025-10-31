import cv2
import numpy as np

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def show_contours(image, contours,title="contour"):
    """
    Display contours on an image
    
    Args:
        image (array): Input image
        contours (list): List of contours
        
    Returns:
        array: Image with contours
    """
    copy = image.copy()
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.drawContours(copy, contours, -1, (0, 255, 0), 4)
    cv2.imshow(title, copy)
    cv2.waitKey(1)
    
def show_ellipse(image, ellipse):
    """
    Display an ellipse on an image
    
    Args:
        image (array): Input image
        ellipse (tuple): Ellipse parameters
        
    Returns:
        array: Image with ellipse
    """
    copy = image.copy()
    cv2.namedWindow('ellipse', cv2.WINDOW_NORMAL)
    cv2.ellipse(copy, ellipse, (0, 255, 0), 4)
    cv2.imshow('ellipse', copy)
    cv2.waitKey(1)
    
def show_image(image, name='image'):
    """
    Display an image
    
    Args:
        image (array): Input image
    """
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)
    cv2.waitKey(1)

def undistort_image(image, camera_matrix, dist_coeffs):
        """
        Undistort an image using calibration parameters
        
        Args:
            image (array): Input image
        
        Returns:
            array: Undistorted image
        """
        if camera_matrix is None:
            raise ValueError("Calibration not performed. Run calibrate() first.")
        
        return cv2.undistort(
            image, 
            camera_matrix, 
            dist_coeffs
        )
        
def load_calibration(calibration_file='camera_calibration.npz'):
    """
    Load calibration parameters from a file
    
    Args:
        calibration_file (str): Path to calibration file
    """
    data = np.load(calibration_file)
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']
    
    return camera_matrix, dist_coeffs

def refine_matrix(h, w, mtx, dist):
    newcameramtx, roi =cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    return newcameramtx, roi

def undistort_image(image, camera_matrix, dist_coeffs, newcameramtx, roi):
    dst = cv2.undistort(image, camera_matrix, dist_coeffs, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    
    return dst

def printProgressBar(value, max, label):
        n_bar = 10 # Size of progress bar
        sup= value/max
        bar = "III" * int(n_bar * sup)
        bar = bar + '-' * int(n_bar * (1-sup))

        return(f"{label} | [{bar:{n_bar}s}] {int(100 * sup)}% ")
        