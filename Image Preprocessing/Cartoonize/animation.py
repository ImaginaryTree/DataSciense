import cv2
def cartoonize(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    

    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 9)

    color = cv2.bilateralFilter(image, 15, 75, 75)
    
    cartoon = cv2.bitwise_and(color, color, mask = edges)
    
    return cartoon