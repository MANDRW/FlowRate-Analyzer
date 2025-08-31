from heic2png import HEIC2PNG
import os


def converter(input_img, output_folder, id, type):
    converted_img = HEIC2PNG(input_img, quality=100)
    converted_img_path = os.path.join(output_folder, os.path.basename(input_img).replace(".HEIC", ".png"))
    converted_img.image.save(converted_img_path)
    os.rename(converted_img_path, os.path.join(output_folder, f"{id}_{type}.png"))

if __name__ == "__main__":
    folder_type_path="close" # far or close
    img_type_path="under" # under or good or over
    input_folder=(f"{folder_type_path}/{img_type_path}")
    output_folder=(f"{folder_type_path}/{img_type_path}_png")
    id = 1 # id of the first image (under)
    type = 100 #100 - under, 010 - good, 001 - over
    os.makedirs(output_folder, exist_ok=True)
    for img in os.listdir(input_folder):
        if img.endswith(".HEIC") and img.startswith("IMG_"):
            input_img = os.path.join(input_folder, img)
            converter(input_img, output_folder,id,type)
            id+=1
    print("DONE")

