import cv2
import numpy as np
import os

i = 1
target = 24
while (i <= target):
    # Load the image
    image = cv2.imread('/workspaces/python/fullImages/fullImage' + str(i) + '.jpg')  # Replace with the name of your image file
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect edges to find the split line
    edges = cv2.Canny(gray, 50, 150)

    # Find horizontal lines using Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    # Initialize split point
    split_point = None

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            if theta > 1.5:  # Horizontal line condition (approximately horizontal)
                split_point = int(abs(rho))  # Extract position
                break

    if split_point is not None:
        print(f"Detected split point at: {split_point}")
    else:
        print("Unable to detect a split point. Using default value.")
        split_point = int(image.shape[0] * 0.75)  # Fallback split point

    # Crop the candlestick chart
    candlestick_chart = image[0:split_point, 0:image.shape[1]]

    # Crop the RSI graph
    
    
    
    #rsi_graph = image[split_point:image.shape[0], 0:image.shape[1]]
    rsi_graph = image[640:image.shape[0], 0:image.shape[1]]





    # Save the cropped images
    cv2.imwrite('cs' + str(i) + '.jpg', candlestick_chart)
    cv2.imwrite('rsi' + str(i)+'.jpg', rsi_graph)
    i+=1

print("Images saved successfully")

################################################################################################################################

################################################################################################################################

################################################################################################################################

#Parse Candle Stick Graph
from PIL import Image
import numpy as np

def keep_selected_colors(image_path, output_path, colors_to_keep, tolerance=30):

    img = Image.open(image_path)
 
    img = img.convert('RGB')

    img_array = np.array(img)
    
    final_mask = np.zeros(img_array.shape[:2], dtype=bool)
    
    for color in colors_to_keep:
        color = np.array(color)

        differences = np.abs(img_array - color)
        total_difference = np.sum(differences, axis=2)
        color_mask = total_difference < tolerance * 3
        final_mask = final_mask | color_mask
    
    mask_3d = np.stack([final_mask] * 3, axis=2)
    

    result = np.where(mask_3d, img_array, 0)
    
    result_img = Image.fromarray(result.astype('uint8'))
    result_img.save(output_path)


if __name__ == "__main__":

    colors_to_keep = [
        (255, 165, 0),  # Orange
        (255, 255, 255) # White
    ]
    
    i = 1

    while i <= target:

        keep_selected_colors(
            'cs' + str(i) + '.jpg',    
            '/workspaces/python/parseData/outputCSChart/outputCS' + str(i) + '.jpg',   
            colors_to_keep,
            tolerance=30
        )
        os.remove('cs' + str(i) + '.jpg')
        i += 1

##############################
#RSI
if __name__ == "__main__":

    colors_to_keep = [
        (255, 184, 28),  # Yellow
        (100, 52, 255)   # Purple
    ]
    
    i = 1

    while i <= target:

        keep_selected_colors(
            'rsi' + str(i) + '.jpg',    
            '/workspaces/python/parseData/outputRSIChart/outputRSI' + str(i) + '.jpg',   
            colors_to_keep,
            tolerance=70
        )
        os.remove('rsi' + str(i) + '.jpg')
        i += 1


