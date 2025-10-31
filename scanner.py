import cv2
import numpy as np
import matplotlib.pyplot as plt
from utilities import load_calibration, ResizeWithAspectRatio, show_contours, show_image, printProgressBar
from turntable import Turntable
from back import Back
from laser import Laser
import time

class Scanner:
    
    def __init__(self, video_path, calibration_file='camera_calibration.npz', output_file="pointcloud.xyz", debug=False):
        self.video_path = video_path
        self.calibration_file = calibration_file
        self.output_file = output_file
        self.debug = debug

    def scan(self):
        # Load calibration parameters
        camera_matrix, dist_coeffs = load_calibration(self.calibration_file)
        
        # Open the video
        cap = cv2.VideoCapture(self.video_path)
        h, w = cap.read()[1].shape[:2]

        output = open(self.output_file, "w")
        
        if(h>w):
            print("Portrait mode")
        
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
        x, y, w, h = roi
        
        cap = cv2.VideoCapture(self.video_path)
        
        total= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        turntable=Turntable(self.debug)
        
        back = Back(self.debug)
        
        laser=Laser(self.debug)
        
        f = camera_matrix[0][0]
        i=0
        
        total_time=0
        
        print("Start Scanning the video")
        
        while cap.isOpened():
            # Read a frame
            ret, frame = cap.read()
            
            if not ret:
                break   
            i+=1
            
            start_time=time.time()
            
            try:
                #unidstort the frame and crop in the region of interest
                frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
                frame = frame[y:y + h, x:x + w]
                
                copy=frame.copy()
                
                #find the center of the turntable and it's R e T parameters
                ellipse, tr, tt =turntable.runPlate(frame, camera_matrix, dist_coeffs)
                
                #find the back marker plane equation and it's R e T parameters
                ab, bb, cb , br, bt=back.run(frame, camera_matrix, dist_coeffs)
                
                #compute the rotation matrix from camera to turntable and from backmarker to camera
                camera_turntable = np.concatenate([np.concatenate([np.array(cv2.transpose(cv2.Rodrigues(tr)[0])),np.array(- cv2.transpose(cv2.Rodrigues(tr)[0]) @ tt)], axis=1),np.array([[0, 0, 0, 1]])], axis=0)
                back_camera = np.concatenate([np.concatenate([cv2.Rodrigues(br)[0], bt], axis=1),np.array([[0, 0, 0, 1]])], axis=0)
                
                #the first point it's the intersection between the back marker y=0 and the laser line
                #first in camera coordinates and then in turntable coordinates
                firstBack=[((-cb - bb * 0) / ab), 0, 0, 1]
                firstCamera =  back_camera @ firstBack
                firstTurn = camera_turntable @ firstCamera
                firstTurn = [firstTurn[0] / firstTurn[3], firstTurn[1] / firstTurn[3],firstTurn[2] / firstTurn[3]]
                
                #the second point it's the intersection between the back marker y=230 and the laser line, 
                #first in camera coordinates and then in turntable coordinates
                secondBack = [((-cb - bb * 230) / ab), 230, 0,1]
                secondCamera = back_camera @ secondBack
                secondTurn = camera_turntable @ secondCamera
                secondTurn = [secondTurn[0] / secondTurn[3], secondTurn[1] / secondTurn[3],
                        secondTurn[2] / secondTurn[3]]
                
                #the third point is found on the flat laser line that lays on the turntable
                third = turntable.findLaser(copy.copy(), ellipse[0]) 
                
                #find camera origin with respect to the turntable
                Rod = cv2.Rodrigues(tr)[0]
                cameraOrigin = -(Rod.T @ tt).flatten()
                
                #find the intersection between a ray that pass through the camera origin and the third point
                #and the turntable plane, this way we can find the point that lays on the laser plane and the turntable
                ip= laser.findPlateIntersection(third, w, h, f, camera_turntable, cameraOrigin)
                
                projected, _ = cv2.projectPoints(
                np.array([
                    ip,
                    [0,0,0],
                    firstTurn,
                    secondTurn,
                ], dtype=np.float32), tr, tt, camera_matrix, np.zeros((4,1)))

                
                cv2.drawMarker(frame, [round(i) for i in projected[0][0]], (0, 255, 0), cv2.MARKER_STAR, 30, 3)
                cv2.drawMarker(frame, [round(i) for i in projected[2][0]], (0, 255, 0), cv2.MARKER_STAR, 30, 3)
                cv2.drawMarker(frame, [round(i) for i in projected[3][0]], (0, 255, 0), cv2.MARKER_STAR, 30, 3)
                
                #now that we have 3 points that lays on the laser plane we can find the plane equation
                laserplane = laser.findPlane(firstTurn, secondTurn, ip)
                
                #we find point in the frame where the laser is detected on the figure 
                figurePoints=laser.findFigure(copy.copy(), ellipse)
                
                for point in figurePoints:
                    # map the point to the camera reference system and then to the turntable reference system
                    pCamera = np.array([point[0] - (w / 2), point[1] - (h / 2), f, 1])
                    pTurntable = camera_turntable @ pCamera  
                    pTurntable = [pTurntable[0] / pTurntable[3], pTurntable[1] / pTurntable[3], pTurntable[2] / pTurntable[3]]  

                    # find the intersection between the laser plane and the line passing through the camera origin
                    #and the point in turntable coordinates
                    interPoint = laser.findFigurePlaneIntersection(laserplane, cameraOrigin, pTurntable)
                    
                    #check if the point is close enough to the turntable center and on a positive z turntable coordinate, if not we discard it
                    if not (-30 < interPoint[0] < 30):
                        continue
                    if not (-30 < interPoint[1] < 30):
                        continue
                    if not (interPoint[2] > 1):
                        continue
                    
                    #draw the point on the image to check correctness of the scan
                    project, _ = cv2.projectPoints(
                    np.array([  
                        interPoint
                    ], dtype=np.float32), tr, tt, camera_matrix, np.zeros((4,1)))
                    
                    # output the object point
                    output.write(f"{interPoint[0]} {interPoint[1]} {interPoint[2]}\n")
                    cv2.drawMarker(frame, [round(i) for i in project[0][0]], (0, 255, 0), cv2.MARKER_CROSS, 3, 3)
            
            except Exception as error:
                print(f"Error in frame {i}: {error}")
                print("Discarding frame")
            
            
            cv2.putText(frame, printProgressBar(i, total, "Scanning"),(50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
            
            show_image(frame, "Scanner")
            output.flush()
            
            end_time=time.time()
            total_time+=end_time-start_time
        
        cv2.destroyAllWindows()
        output.close()
        print("Scanning completed")
        print(f"Total time: {total_time} s")
        print(f"Average time per frame: {total_time/total} s")
            
if __name__ == '__main__':
    
    cat='G3DCV_turntable_scanner_data\\cat.mov'
    cube='G3DCV_turntable_scanner_data\\cube.mov'
    ball='G3DCV_turntable_scanner_data\\ball.mov'
    
    
    debug={
        "ellipses": True,
        "inliers": True,
        "turntable": True,
        "back laser": True,
        "turntable laser": True,
    }
    
    scanner = Scanner(cat, debug=debug, output_file="cat.xyz")
    scanner.scan()