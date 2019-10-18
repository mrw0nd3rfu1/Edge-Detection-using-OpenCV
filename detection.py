import matplotlib.pylab as plt
import cv2
import numpy as np

#defines reion of interest by masking the original image
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

#function takes original image and draw the lines on it
def draw_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1,y1), (x2,y2), (0, 255, 0), thickness=10)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

#use below code if want to use image for image detection
#image = cv2.imread('test.jpg')                                         
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


#function checks frame by frame of a video and detects edges
def process(image):
    height = image.shape[0]
    width = image.shape[1]
    #region of interest depends upon the solution you want
    region_of_interest_vertices = [
        (0, height),
        (width/2, height/2),
        (width, height)
    ]
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #using canny edge method to detect edges in a frame
    canny_image = cv2.Canny(gray_image, 100, 120)
    cropped_image = region_of_interest(canny_image,
                    np.array([region_of_interest_vertices], np.int32),)
    #using hough line transfor to get the lines
    lines = cv2.HoughLinesP(cropped_image,
                            rho=2,
                            theta=np.pi/180,
                            threshold=50,
                            lines=np.array([]),
                            minLineLength=40,
                            maxLineGap=100)
    image_with_lines = draw_the_lines(image, lines)
    return image_with_lines

 #use this if we want a video to detect edges
cap = cv2.VideoCapture('test.mp4')                                 

while cap.isOpened():
    ret, frame = cap.read()
    frame = process(frame)
    cv2.imshow('frame', frame)
    #press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):                           
        break

cap.release()
cv2.destroyAllWindows()
