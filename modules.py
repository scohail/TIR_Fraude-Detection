import supervision as sv
import cv2
from PIL import Image
import os
from inference import get_model
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
from collections import Counter
import easyocr










# Predefined list of colors with their RGB values

colors_dict = {
    'Red': [255, 0, 0],
    'salmon': [250, 128, 114],
    'orange_red': [255, 69, 0],
    'dark_red': [139, 0, 0],


    'Green': [0, 120, 0],
    'dark_green': [0, 128, 0],
    'sea_green': [46, 139, 87],

    'Blue': [0, 0, 255],
    'sky_blue': [135, 206, 235],
    'dark_blue': [0, 0, 139],
    'navy': [0, 0, 128],


    'Yellow': [255, 255, 0],
    'Cyan': [0, 255, 255],
    'Black': [0, 0, 0],
    'White': [255, 255, 255],
    'Gray': [128, 128, 128],


    'Orange': [255, 140, 0],
    'gold': [255, 215, 0],
    
    # Add more colors if needed
}

# Convert the dictionary to a list of colors and names
color_names = list(colors_dict.keys())
color_values = np.array(list(colors_dict.values()))

# Build a KDTree for fast nearest-neighbor search
kdtree = KDTree(color_values)














# Function to detect the truck in an image
def detect_Trailer(image):

    # load a pre-trained yolov8n model with the API key
    api_key = "LkSWsyPx89eiIgyQFZ0b"

    #model = get_model(model_id="containerdetection-76m5b/1", api_key=api_key)
    
    model = get_model(model_id="vehicle-classification-v2/1", api_key=api_key)    
    # run inference on our chosen image, image can be a url, a numpy array, a PIL image, etc.


    print("Model loaded successfully")

    print("detecting truck in the image...")
    results = model.infer(image)[0]
    # load the results into the supervision Detections api
    detections = sv.Detections.from_inference(results)

    # create supervision annotators
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    mask_annotator = sv.MaskAnnotator()
    

    # annotate the image with our inference results
    

    annotated_image = bounding_box_annotator.annotate(
        scene=image, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)
    # annotated_image = mask_annotator.annotate(
    #     scene=image, detections=detections)
    
    print("Truck detected successfully")

    #annotated_image.save("masked_image.png")
    
    #save the masked image 



    return annotated_image, detections



# Function to crop the detected truck and detect the colors in the cropped image
def Crop_Trailer(image, detections):

    print("Cropping the detected truck...")

    # Assuming there is only one detection and you want to crop the masked area
    mask = detections.mask[0]  # Get the mask of the first detection

    # Convert the mask to a binary mask (True/False)
    binary_mask = mask.astype(bool)

    # Find the bounding box of the masked area
    coords = np.argwhere(binary_mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop the original image using the bounding box coordinates
    cropped_image = image.crop((x_min, y_min, x_max, y_max))
    

    color_names=extract_top_colors_from_pil_image(cropped_image, num_colors=2)
    print("Colors detected in the image:", color_names)

    
    

    # Convert the cropped image to a NumPy array
    cropped_image_np = np.array(cropped_image)

    # Create a white background image of the same size as the cropped image
    white_background = np.ones_like(cropped_image_np) * 255

    # Apply the mask to the cropped image to get the exact masked part
    masked_cropped_image = np.where(binary_mask[y_min:y_max, x_min:x_max, np.newaxis], 
                                    cropped_image_np, 
                                    white_background)

    # Convert the result back to an Image object
    masked_cropped_image_pil = Image.fromarray(masked_cropped_image.astype(np.uint8))


    masked_cropped_image_pil.save("masked_image.png")

    brand_name = detect_brand(masked_cropped_image)

    return masked_cropped_image_pil , brand_name




# Function to detect the brand name in an image
def detect_brand(image):   
    # Initialize the reader with English language
    reader = easyocr.Reader(['en'], gpu=True)

    # Read text from the image
    result = reader.readtext(image)
    brand_name = None

    # post-process the result to extract the brand name depending on the use case

    for i in range (len(result)):

        if result[i][1] == 'CAP':
            brand_name = 'CAP'
            break

        elif result[i][1] == 'SJL':
            brand_name = 'SJL'
            break
        
        elif result[i][1] == "'Transmodal":
            brand_name = 'Transmodal'
            break

        elif result[i][1] == 'Lodisna':
            brand_name = 'Lodisna'
            break


        elif result[i][1] == 'DACHSER':
            brand_name = 'DACHSER'
            break

        elif result[i][1] == 'marcotran Com':
            brand_name = 'marcotran'
            break
        
        elif result[i][1] == 'XPOLogistics' or result[i][1] == 'XPOLogstk s':
            brand_name = 'XPOLogistics'
            break

        elif result[i][1] == 'GTM':
            brand_name = 'GTM'
            break

    if brand_name == None:   
        brand_name = [result[i][1] for i in range(len(result))]
        
    
    return brand_name


# Function to detect the color of an image
def get_color_name(rgb_color):
    # Find the nearest color in the KDTree
    distance, index = kdtree.query(rgb_color)
    return color_names[index]



# Function to extract the top colors from a PIL image
def extract_top_colors_from_pil_image(pil_image, num_colors=3):
    # Convert PIL image to NumPy array
    image_array = np.array(pil_image)
    
    # Ensure the image is in RGB format
    if image_array.shape[-1] == 4:  # If the image has an alpha channel
        image_array = image_array[:, :, :3]  # Drop alpha channel
    elif image_array.shape[-1] == 1:  # If the image is grayscale
        image_array = np.stack([image_array]*3, axis=-1)  # Convert to RGB

    # Reshape the image to be a list of pixels
    pixels = image_array.reshape((-1, 3))
    
    # Perform KMeans clustering to find the most dominant colors
    kmeans = KMeans(n_clusters= 2*num_colors)  # More clusters to get a more detailed result
    kmeans.fit(pixels)
    
    # Get the cluster centers and their frequencies
    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    color_frequencies = dict(zip(unique, counts))
    
    # Sort colors by frequency
    sorted_colors = sorted(color_frequencies.keys(), key=lambda x: color_frequencies[x], reverse=True)
    
    # Get the top colors
    top_colors = kmeans.cluster_centers_[sorted_colors[:num_colors]]
    
    # Map to color names
    top_color_names = [get_color_name(color) for color in top_colors]
    
    return top_color_names










'''


For testing the functions


image_file = "containers/SJL1.jpg"


# Check if the image file exists
if not os.path.exists(image_file):
    raise FileNotFoundError(f"The image file {image_file} does not exist.")

# Load the image using OpenCV
image = cv2.imread(image_file)

# Check if the image was loaded successfully
if image is None:
    raise ValueError(f"Failed to load the image file {image_file}.")

# Convert the NumPy array to a PIL image
image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Detect containers in the image
annotated_image, detections = detect_Trailer(image_pil)

# Crop and detect colors in the detected containers
cropped_image, brand_name  = Crop_Trailer(annotated_image, detections)




# Detect color in the cropped image
#color_detected_image = detect_color(cropped_image)



# Display the annotated image with container detections
sv.plot_image(annotated_image)


# Convert the color-detected image back to PIL for displaying

sv.plot_image(cropped_image)

print("Brand name detected in the image:", brand_name)

'''