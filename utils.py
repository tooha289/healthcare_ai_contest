import json
import shutil
import os

def move_files(json_filepath, source_dir, destination_dir_true, destination_dir_false):
    with open(json_filepath, 'r') as json_file:
        data = json.load(json_file)

    image_filepath = data["image_filepath"]
    tooth_list = data["tooth"]

    decayed_flag = any(tooth["decayed"] for tooth in tooth_list)

    if decayed_flag:
        destination_dir = destination_dir_true
    else:
        destination_dir = destination_dir_false

    image_filename = os.path.basename(image_filepath)
    source_image_path = os.path.join(source_dir, image_filename)
    destination_image_path = os.path.join(destination_dir, image_filename)

    # Check if the source image file exists
    if os.path.exists(source_image_path):
        # Move the file
        shutil.move(source_image_path, destination_image_path)
        print(f"Moved {image_filename} from {source_dir} to {destination_dir}")
    else:
        print(f"Skipping {image_filename} because it doesn't exist in the source directory.")

if __name__ == "__main__":
    data_type = "train_data"
    jsons_directory_path = f"../Dataset/{data_type}/json/"
    source_dir = f"../Dataset/{data_type}/image"
    destination_dir_true = f"../Dataset/{data_type}/true"
    destination_dir_false = f"../Dataset/{data_type}/false/"

    for filename in os.listdir(jsons_directory_path):
        if filename.endswith(".json"):
            json_filepath = os.path.join(jsons_directory_path, filename)
            move_files(json_filepath, source_dir, destination_dir_true, destination_dir_false)
