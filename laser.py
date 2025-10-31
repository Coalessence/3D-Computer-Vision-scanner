import numpy as np
import cv2
from utilities import show_ellipse, show_image

class Laser:
    
    def __init__(self, debug=False):
        self.debug = debug
    
    def findPlane(self, point1, point2, point3):
        """
        Find the equation of a plane passing through three points.
        Returns normalized plane coefficients.
        """
        
        vector1 = np.array(point2) - np.array(point1)
        vector2 = np.array(point3) - np.array(point1)
        
        normal_vector = np.cross(vector1, vector2)
        
        norm = np.linalg.norm(normal_vector)
        if norm < 1e-6:  # Check for degenerate case (collinear points)
            raise ValueError("Points are collinear, plane is undefined")
            
        normal_vector = normal_vector / norm
        
        d = -np.sum(point1 * normal_vector)
        
        return np.array([normal_vector[0], normal_vector[1], normal_vector[2], d])
    

    def findFigure(self, frame, ellipse):
        
        # blur image with a bilateral filter so we keep edges sharp
        blurred = cv2.bilateralFilter(frame, 5, 50, 125)  

        hsv = cv2.cvtColor(blurred[round(ellipse[0][1] - ellipse[1][0] / 2): round(ellipse[0][1]+ellipse[1][0] / 2),
            round(ellipse[0][0] - ellipse[1][1] / 2): round(ellipse[0][0] + ellipse[1][1] / 2),:], cv2.COLOR_BGR2HSV)

        # red laser mask ranges
        lower_red = cv2.inRange(hsv, (0, 70, 225), (20, 255, 255))
        higher_red = cv2.inRange(hsv, (160, 60, 225), (179, 255, 255))

        laser = lower_red | higher_red

        figure = []
        
        points=cv2.findNonZero(laser)
        y_to_min_x = {}

        for point in points:
            x, y = point[0]
            # If we haven't seen this y before, or if this x is smaller than what we've seen
            if y not in y_to_min_x or x < y_to_min_x[y]:
                y_to_min_x[y] = x
        
        #we take only the leftmost point because otherwise the figure is not sharp and the point cloud is too dense
        leftmost_points = [[x + ellipse[0][0] - ellipse[1][1] / 2, y+ellipse[0][1] - ellipse[1][0] / 2] for y, x in y_to_min_x.items()]
        
        return leftmost_points
        

    def findPlateIntersection(self, point, width, height, focal_length, camera_turntable, cameraOrigin):
        """
        Compute plate intersection point using plane-line intersection
        """
        # Create image point in camera coordinates
        cameraPoint = np.array([
            point[0] - width/2,
            point[1] - height/2,
            focal_length,
            1.0
        ])
        
        # Transform to turntable coordinates
        turntablePoint = camera_turntable @ cameraPoint
        turntablePoint = turntablePoint[:3] / turntablePoint[3]
    
        # Define plane (z = 0)
        pn = np.array([0, 0, 1])
        pp = np.array([0, 0, 0])
        
        # Calculate intersection
        ray = turntablePoint - cameraOrigin
        d = np.dot(pn, pp - cameraOrigin) / np.dot(pn, ray)
        intersection = cameraOrigin + d * ray
        
        return intersection
    
    def findFigurePlaneIntersection(self, plane, point1, point2):
        """
        Find the intersection between a plane and a line passing through the camera origina and a point on the figure
        """
        ray = point2 - point1  # Direction vector
        normal = plane[:3]     # Get plane normal vector
        
        # Check if line is parallel to plane
        denom = np.dot(normal, ray)
        if abs(denom) < 1e-6: 
            return None
            
        # Calculate distance ray to intersection
        # First calculate: -(normal·point1 + d)
        # Then divide by denominator to get parameter t that tells us how far the ray to go
        d = -(np.dot(normal, point1) + plane[3]) / denom
        
        # Using parametric line equation: point = point1 + t*ray
        intersection = point1 + d * ray
    
        return intersection
    