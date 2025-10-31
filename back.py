import cv2
import numpy as np
from utilities import show_contours, show_ellipse, show_image
import math

class Back:

    def __init__(self, debug=False):
        self.debug = debug
    
    def addCopy(self, frame):
        self.copy = frame.copy()
    
    def frameProcessing(self, frame):
            
            gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            
            kernel = np.ones((5,5), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                
            return binary

    def detectRectangle(self, frame):
    
        frame= self.frameProcessing(frame)
        # Find contours of the marker
        contours, hier = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        h, w = frame.shape[:2]
        
        
        candidates = []
        
        for i, contour in enumerate(contours):
            
            #i know the back marker i want is the inner one so i can check the hierarchy
            if hier[0][i][3]==i-1 and hier[0][i][2]==-1 and hier[0][i][1]==-1:
                epsilon = 0.01 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                area=cv2.contourArea(approx)
                
                if len(approx) == 4 and area>5000 and approx[0][0][0]<300 and approx[0][0][1]<400:
                    candidates.append(approx)
        
        return candidates[0]    
            
        
    def run(self, frame, camera_matrix, dist_coeffs):
        
        self.addCopy(frame)
        
        try:
            rec=self.detectRectangle(self.copy)
            
            a, b, c, d = rec
            a = a[0]
            b = b[0]
            c = c[0]
            d = d[0]
            
            backP=[a,b,c,d]
            recP=[[0, 0],[130, 0],[130, 230],[0, 230]]
            
            
            #having the coordinates in pixel of the four corners of the back marker i can find the homography and recover the pose
            H = cv2.findHomography(np.array(backP), np.array(recP))
            
            recP = np.array(recP, dtype=np.float32)
            recP = np.append(recP, np.zeros((len(recP), 1)), axis=1)
            
            res, r, t = cv2.solvePnP(np.array(recP, dtype=np.float32), np.array(backP, dtype=np.float32), camera_matrix, np.zeros((4,1)), flags=cv2.SOLVEPNP_IPPE)
            
            hFrame = cv2.warpPerspective(frame, H[0], (480, 960))

            #in the homography warped frame i find the laser line
            a, b, c = self.find_laser_line(hFrame)
            
            
            projec, _ = cv2.projectPoints(np.array([
            [0, 0, 0],
            [130, 0, 0],
            [130, 230, 0],
            [0, 230, 0],
            ], dtype=np.float32), r, t, camera_matrix, np.zeros((4,1)))
            cv2.line(frame, [round(projec[0][0][0]),round(projec[0][0][1])], [round(projec[1][0][0]),round(projec[1][0][1])], (0, 0, 255), 3)
            cv2.line(frame, [round(projec[1][0][0]),round(projec[1][0][1])], [round(projec[2][0][0]),round(projec[2][0][1])], (0, 0, 255), 3)
            cv2.line(frame, [round(projec[2][0][0]),round(projec[2][0][1])], [round(projec[3][0][0]),round(projec[3][0][1])], (0, 0, 255), 3)
            cv2.line(frame, [round(projec[0][0][0]),round(projec[0][0][1])], [round(projec[3][0][0]),round(projec[3][0][1])], (0, 0, 255), 3)
            cv2.drawMarker(frame, [round(projec[0][0][0]),round(projec[0][0][1])], (0, 255, 0), cv2.MARKER_CROSS, 5, 5)
            cv2.drawMarker(frame, [round(projec[1][0][0]),round(projec[1][0][1])], (0, 255, 0), cv2.MARKER_CROSS, 5, 5)
            cv2.drawMarker(frame, [round(projec[2][0][0]),round(projec[2][0][1])], (0, 255, 0), cv2.MARKER_CROSS, 5, 5)
            cv2.drawMarker(frame, [round(projec[3][0][0]),round(projec[3][0][1])], (0, 255, 0), cv2.MARKER_CROSS, 5, 5)
        
        except Exception as error:
            raise error
        
        return a,b,c, r, t
        
    def find_laser_line(self, frame):
        """
        Find the laser line on the rectangular marker given its homography
        Returns:
            tuple: (a, b, c) parameters of the line equation ax + by + c = 0
        """
        
        
        #Crop the frame wrt the marker, extract red channel and threshold to find laser line
        frame = frame[0:230, 0:130]
        red = frame[:, :, 2].copy()
        _, laser_mask = cv2.threshold(red, 200, 255, cv2.THRESH_BINARY)
        
        
        points = cv2.findNonZero(laser_mask)      
        if points is None or len(points) < 2:
            raise Exception("Not enough laser points detected on rectangular marker")
        
        #use fitLine to find the line equation from non zero laser points
        # Returns the normalized vector [vx, vy] and point on the line [x0, y0] the equation is: (y-y0)/(x-x0) = vy/vx
        [vx, vy, x0, y0] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        
        if self.debug["back laser"]:
            debug_img = frame.copy()
            lefty = int((-x0 * vy/vx) + y0)
            righty = int(((130 - x0) * vy/vx) + y0)
            cv2.line(debug_img, (0,lefty), (130,righty), (0,255,0), 2)
            show_image(debug_img, "back Laser")
        
        # Convert to ax + by + c = 0 form
        # vy*x - vx*y + (vx*y0 - vy*x0) = 0
        a = vy[0] 
        b = -vx[0] 
        c = (vx[0]*y0[0] - vy[0]*x0[0])
        
        return a, b, c
            