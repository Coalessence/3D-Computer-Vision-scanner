import cv2
import numpy as np
from utilities import show_contours, show_ellipse, show_image
import random
import math

class Turntable:
    
    def __init__(self, debug=False):
        self.debug = debug
        self.necklace = 'YWMBMMCCCYWBMYWBYWBC'
    
    def addCopy(self, frame):
        self.copy = frame.copy()
    
    def frameProcessing(self, frame):
        
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel)
        
            
        return binary
    
    def detect_turntable_markers(self, frame):
        """
        Detect the markers of a turntable using contour detection and ellipse fitting
        """

        debugCopy = frame.copy()
        
        frame = self.frameProcessing(frame)
        
        centers = []
        
        (h, w) = frame.shape[:2]
        contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        for i, contour in enumerate(contours):
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                
                center, axes, angle = ellipse
                
                # Store ellipse centers only if small enough,not too close to the sides and in the lower half of the image (y is start from the top)
                if axes[0] > 15 and axes[0] < 70 and axes[1]>15 and axes[1]<70 and center[0]>30 and center[0]<w-30 and center[1]>h/2:
                    duplicate = False
                    for c in centers:
                        if np.linalg.norm(np.array(c) - np.array(center)) < 50:
                            duplicate = True
                            break
                    if not duplicate:
                        centers.append((round(center[0]), round(center[1])))
                        if self.debug['ellipses']:
                            cv2.ellipse(debugCopy, ellipse, (0, 255, 0), 4)
            
        if self.debug['ellipses']:
            show_image(debugCopy, "ellipse_contours")        
    
        return centers
    
    def findInliers(self, points, ellipse, threshold):
        inliers = []
        for point in points:
            dist = abs(self.ellipse_distance(point, ellipse)) 
            if dist < threshold:
                inliers.append(point)
            
        return inliers
    
    def point_to_polar(self, point, center):
        """
        Convert a point's coordinates to polar coordinates (r, theta)
        """
        dx = point[0] - center[0]
        dy = point[1] - center[1]

        radius = round(np.linalg.norm((dx, dy)))
        angle =round(math.degrees(math.atan2(dy, dx)))

        if angle < 0:
            angle += 360

        return (radius, angle)
    
    def detectColor(self, image, point):
        """
        Detect the color of a specific point in an image using HSV color space.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[point[1], point[0]]
        
        # Hue ranges from 0-179 in OpenCV
        range = {
            'C': ((75, 100, 100), (119, 255, 255)),     
            'Y': ((5, 100, 100), (60, 255, 255)),      
            'M': ((120, 100, 100), (175, 255, 255)),    
            'W': ((0, 0, 170), (179, 90, 255)),         
            'B': ((0, 0, 0), (179, 255, 70))            
        }
        
        for color, (lower, upper) in range.items():
            if all([
                lower[0] <= h <= upper[0],
                lower[1] <= s <= upper[1],
                lower[2] <= v <= upper[2]
            ]):
                return color
        
        print(f"Warning: Color not found for point {point} with HSV values {h, s, v}")
        return 'E'
    
    def ransac_ellipse(self, points):
        """
        Fit and ellipse using RANSAC method
        """
        
        if len(points) < 5:
            print("Not enough points to fit an ellipse")
            return None
        
        
        iterations = 100
        best_ellipse = None
        max_inliers = 0
        threshold = 5
        
        # RANSAC ellipse fitting, it find the best ellipse with the most inliers given a sample of 5 points to fit the ellipse
        for _ in range(iterations):
            sample = random.sample(points, 5)
            
            ellipse = cv2.fitEllipse(np.array(sample))

            inliers = []
            
            inliers = self.findInliers(points, ellipse, threshold)
                    
            if len(inliers) > max_inliers:              
                max_inliers = len(inliers)
                best_ellipse = ellipse
            
        if self.debug["ellipses"] and best_ellipse is not None:
            copy=self.copy.copy()
            cv2.ellipse(copy, best_ellipse, (0, 255, 0), 4)
            show_image(copy, "Best ellipse ")
        
        return best_ellipse

    #ellipse distance from a point, it transform the ellipse in a polygon compute distance with opencv functions
    def ellipse_distance(self, point, ellipse):
        poly = cv2.ellipse2Poly((round(ellipse[0][0]), round(ellipse[0][1])),
                           (round(ellipse[1][0] / 2), round(ellipse[1][1] / 2)),
                           round(ellipse[2]), 0, 360, 10)
        
        return cv2.pointPolygonTest(poly, point, True)
    
    #I first tried this method but it was not working properly, giving a distance that scaled with ellipse size
    def ellipse_distance1(self, point, ellipse):
        """
        Calculate the distance of a point from an ellipse.
        """
        (center_x, center_y), (major_axis, minor_axis), angle = ellipse
        
        translated_point = np.array(point) - np.array([center_x, center_y])
        
        angleR = np.deg2rad(-angle)
        rotation_matrix = np.array([
        [np.cos(angleR), -np.sin(angleR)],
        [np.sin(angleR),  np.cos(angleR)]
        ])
        rotated_point = rotation_matrix @ translated_point
        # Normalize point coordinates by ellipse axes
        x, y = rotated_point
        norm_x = x / (major_axis / 2)
        norm_y = y / (minor_axis / 2)
        
        distance = np.sqrt(norm_x**2 + norm_y**2) - 1
        
        return abs(distance)
    
    def polarAdj(self, a, b):
        if abs((a["polar"][1]-b["polar"][1])+360)%360<=26:
            return True 
        return False
    

    def necklaceOrder(self, mElements):
        temp=list()
        temp=[-1]*len(mElements)
        n=self.necklace+("YWM")
        l=len(mElements)
        for i in range(l):
            last=-1
            flag=True
            string=""
            
            #if not temp[i]==-1:
            #    continue
            
            for j in range(4):
                if mElements[(i+j)%l]["color"]=='E':
                    flag=False
                    break
                    
                string+=mElements[(i+j)%l]["color"]
                if not self.polarAdj(mElements[(i+j)%l], mElements[(i+j+1)%l]):
                    flag=False
                    break
            
            if flag:
                idx=n.find(string)
                temp[i]=idx
                temp[(i+1)%l]=(idx+1)%20
                temp[(i+2)%l]=(idx+2)%20
                temp[(i+3)%l]=(idx+3)%20
                
                for k in range(3, l):
                    if self.polarAdj(mElements[(i+k)%l], mElements[(i+k+1)%l]):
                        temp[(i+k+1)%l]=(idx+k+1)%20
                    else:
                        break
        
                last=idx
        
        return temp
                
    
    def runPlate(self,frame, camera_matrix, dist_coeff):
        
        self.addCopy(frame)
        
        try:
            #it finds all the markers on the turntable
            points=self.detect_turntable_markers(frame)
            
            #from the given markers it finds the best ellipse that fits the markers and keeps only the inliers
            ellipse=self.ransac_ellipse(points)
            points=self.findInliers(points, ellipse, 5)
            
            
            if self.debug["inliers"]:
                copy=self.copy.copy()
                for point in points:  
                    cv2.drawMarker(copy, np.array(point).astype(int), (0, 127, 255), cv2.MARKER_STAR)
                    cv2.line(copy, np.array(point).astype(int), np.array(ellipse[0]).astype(int), color=(0, 127, 255))
                show_image(copy, "inliers")
            
            mElements=[]
            
            
            #for each marker it finds the color and the polar coordinates
            for p in points:
                color=self.detectColor(frame, p)
                polar=self.point_to_polar(p, ellipse[0])
                
                mElements.append({
                    'color': color,
                    'polar': polar, 
                    'cartesian': p
                })
            
            mElements.sort(key=lambda x: x['polar'][1], reverse=True)
            
            if len(mElements) > 20:
                raise Exception("Too many markers detected")
            
            #order the markers according to index number detection
            #it finds a known sequence of 4 markers and then it orders the rest of the adjacent ones without worrying about color
            indexes=self.necklaceOrder(mElements)
            
            plateP=[]
            eleP=[]
            
            for i in range(len(indexes)):
                if indexes[i]==-1:
                    continue
                
                alpha=math.radians(18*indexes[i])
                plateP.append([75*np.cos(alpha), 75*np.sin(alpha)])
                eleP.append(mElements[i]["cartesian"])
                cv2.putText(frame, f"{indexes[i]}{mElements[i]['color']}", mElements[i]["cartesian"], cv2.FONT_ITALIC, 1, (255, 0, 0), 1, cv2.LINE_AA)
            
            
            if len(plateP) < 6 or len(eleP) < 6:
                raise Exception("Not enough points to calculate turntable pose")
                
            
            plateP = np.array(plateP, dtype=np.float32)
            plateP = np.append(plateP, np.zeros((len(plateP), 1)), axis=1)
            
            ret, r, t = cv2.solvePnP(np.array(plateP, dtype=np.float32), np.array(np.array(eleP, dtype=np.float32), dtype=np.float32),camera_matrix, np.zeros((4,1)), flags=cv2.SOLVEPNP_IPPE)
            
            self.r=r
            self.t=t
            
            #project axis on the plate
            projec, _ = cv2.projectPoints(np.array([
            [0, 0, 0],
            [75, 0, 0],
            [0, 75, 0],
            [0, -75, 0],
            [-75, 0, 0],
            ], dtype=np.float32), r, t, camera_matrix, np.zeros((4,1)))
            cv2.line(frame, [round(projec[0][0][0]),round(projec[0][0][1])], [round(projec[1][0][0]),round(projec[1][0][1])], (255, 255, 255), 3)
            cv2.line(frame, [round(projec[0][0][0]),round(projec[0][0][1])], [round(projec[2][0][0]),round(projec[2][0][1])], (255, 255, 255), 3)
            cv2.line(frame, [round(projec[0][0][0]),round(projec[0][0][1])], [round(projec[3][0][0]),round(projec[3][0][1])], (255, 255, 255), 3)
            cv2.line(frame, [round(projec[0][0][0]),round(projec[0][0][1])], [round(projec[4][0][0]),round(projec[4][0][1])], (255, 255, 255), 3)
            cv2.drawMarker(frame, [round(projec[0][0][0]),round(projec[0][0][1])], (255, 255, 255), cv2.MARKER_CROSS)
        
        except Exception as error:
            raise error
        
        return ellipse, r, t
                    
    def findLaser(self, frame, center):
        """
        Find a laser point on the plate image
        :param frame: the frame
        :param center: the plate center on the image
        :param debug: debug mode flag
        :return: the coordinates of the point
        """
        # crop frame below ellipse center where laser line is flat
         
        cropped = frame[
                round(center[1]) + 100:round(center[1]) + 200,
                round(center[0]) - 50:round(center[0]) + 150,
                :]
        
        red = cropped[:, :, 2].copy()
        _, laser_mask = cv2.threshold(red, 225, 255, cv2.THRESH_BINARY)
        
        
        if self.debug["turntable laser"]:
            show_image(laser_mask, "plate laser")
        
        points = cv2.findNonZero(laser_mask)      
        if points is None or len(points) < 2:
            raise Exception("Not enough laser points detected on rectangular marker")
        
        # Returns the normalized vector [vx, vy] and point on the line [x0, y0] the equation is: (y-y0)/(x-x0) = vy/vx
        [vx, vy, x0, y0] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        
        point1 = np.array([round(x0[0]+center[0]-50), round(y0[0]+center[1]+100)])
            
        return point1
        
