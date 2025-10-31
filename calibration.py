import cv2
import numpy as np
import matplotlib.pyplot as plt
from utilities import ResizeWithAspectRatio, printProgressBar

class CameraCalibration:

    def __init__(self, pattern_size, square_size):
        
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.pattern_size = pattern_size
        self.square_size = square_size
        self.object_points = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        self.object_points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        self.object_points *= square_size
        self.object_points_list = []
        self.image_points_list = []
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
    
    def loadVideoByFrames(self, video_path):
        print("Wait a moment loading video from: ", video_path)
        cap = cv2.VideoCapture(video_path)
        total= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        frame_list = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            else:
                frame_list.append(frame)
            frame_count += 1
        cap.release()
        return frame_list, frame_count

    # Load calibration images and find chessboard corners
    def findCorners(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)

        if ret:
            self.object_points_list.append(self.object_points)
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), self.criteria)
            self.image_points_list.append(corners2)

        self.image_size = gray.shape[::-1]

    def calibrateCamera(self):
        # Calibrate camera
        ret, self.camera_matrix, self.dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(self.object_points_list, self.image_points_list, self.image_size, None, None)
        print("RMS of the calibration:", ret)
        
        mean_error = 0
        for i in range(len(self.object_points_list)):
            imgpoints2, _ = cv2.projectPoints(self.object_points_list[i], rvecs[i], tvecs[i], self.camera_matrix, self.dist_coeffs)
            error = cv2.norm(self.image_points_list[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error
        
        print("Mean reprojection error: ", mean_error/len(self.object_points_list))
        
        return self.camera_matrix, self.dist_coeffs, rvecs, tvecs, mean_error/len(self.object_points_list)
    
    def save_calibration(self, output_file='camera_calibration.npz'):
        """
        Save calibration parameters to a file
        
        Args:
            output_file (str): Path to save calibration data
        """
        
        print("Saving calibration to ", output_file)
        np.savez(output_file, camera_matrix=self.camera_matrix, dist_coeffs=self.dist_coeffs)
    
    def run(self, video_path, frame_number):
        frames, frame_count = self.loadVideoByFrames(video_path)
        
        #takes only frame_number frames from different video part
        frames = frames[::frame_count//frame_number]
        i=0
        for frame in frames:
            if __debug__:
                print("Processing frame ", i)
                i+=1
            self.findCorners(frame)
        camera_matrix, dist_coeff, r, t, rms =self.calibrateCamera()
        
        h, w = frames[0].shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeff, (w, h), 1, (w, h))
        new = cv2.undistort(frames[0], camera_matrix, dist_coeff, None, newcameramtx)
        x, y, w, h = roi
        new = new[y:y+h, x:x+w]
        cv2.namedWindow("undistort", cv2.WINDOW_NORMAL)
        cv2.imshow('undistort', new)
        cv2.waitKey(0)
        
        print("Calibration done")
        print("Camera matrix: ", camera_matrix)
        print("Distortion coefficients: ", dist_coeff)
        

if __name__ == '__main__':
    
    cc=CameraCalibration((9,6), 1)
    cc.run('./G3DCV_turntable_scanner_data/calibration.mov', 15)
    cc.save_calibration('camera_calibration.npz')
