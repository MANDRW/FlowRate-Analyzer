from heic2png import HEIC2PNG
import os
import shutil


def converter(input_img, output_folder, id, type):
    converted_img = HEIC2PNG(input_img, quality=100)
    converted_img_path = os.path.join(output_folder, os.path.basename(input_img).replace(".HEIC", ".png"))
    converted_img.image.save(converted_img_path)
    os.rename(converted_img_path, os.path.join(output_folder, f"{id}_{type}.png"))

if __name__ == "__main__":
    folder_type_path="far" # far or close
    img_type_path="good" # under or good or over
    input_folder=(f"{folder_type_path}/{img_type_path}")
    output_folder=(f"{folder_type_path}/{img_type_path}_png")
    id = 201 # id of the first image (1 - under, 201 - good, 401 - over)
    type = "010" #100 - under, 010 - good, 001 - over
    if(not os.path.exists(output_folder)):
        os.makedirs(output_folder, exist_ok=True)
    else:
        shutil.rmtree(output_folder)
        os.makedirs(output_folder, exist_ok=True)
    if(not os.path.exists(input_folder)):
        print("Folder doesn't exists")
        exit()
    for img in os.listdir(input_folder):
        if img.endswith(".HEIC") and img.startswith("IMG_"):
            input_img = os.path.join(input_folder, img)
            converter(input_img, output_folder,id,type)
            print(id)
            id+=1
    print("DONE")
