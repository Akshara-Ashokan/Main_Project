import os
import cv2
import main

# answer of the omr sheet in order
answers = ['b', 'd', 'a', 'a', 'e']
negative_marking = True
questions = 5
choices = 5

# input and output folder paths
input_folder_path = './OMR_Sheets'
output_folder_path = './OMR_Answers'

# Ensure the folder path exists
if os.path.exists(input_folder_path):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Loop through all images in the folder
    for filename in os.listdir(input_folder_path):
        # Check if the file is an image (you may want to add more file type checks if needed)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Construct the full path to the image
            image_path = os.path.join(input_folder_path, filename)

            # calling the function in main to get the answers marked in the image
            img, score = main.get_answers(image_path, answers, negative_marking, questions, choices)

            new_filename = f"{os.path.splitext(filename)[0]}_{score}%{os.path.splitext(filename)[1]}"

            # Construct the full path to the output image
            output_image_path = os.path.join(output_folder_path, new_filename)

            # Save the processed image using cv2.imwrite
            cv2.imwrite(output_image_path, img)
        else:
            print(f"Skipped non-image file: {filename}")

else:
    print(f"The folder path '{input_folder_path}' does not exist.")
