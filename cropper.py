import cv2
import os

def crop( image, crop_ratio=0.5):
    h, w, _ = image.shape
    new_h, new_w = int(h * crop_ratio), int(w * crop_ratio)
    left = max(0, (w - new_w) // 2)
    top = max(0, (h - new_h) // 2)
    right = left + new_w
    bottom = top + new_h
    return image[top:bottom, left:right]


def process_images(input_folder, output_folder, crop_ratio=0.5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        if filename.lower().endswith(('.png')):
            image = cv2.imread(input_path)
            if image is None:
                print(f"Failed to load image: {input_path}")
                continue

            cropped_image = cv2.rotate(image,cv2.ROTATE_180)#crop(image, crop_ratio)

            number= filename.split("_")[0]
            number=int(number)+1200#600
            new_filename = f"{number}_{'_'.join(filename.split('_')[1:])}"
            output_path = os.path.join(output_folder, new_filename)

            cv2.imwrite(output_path, cropped_image)
            print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    input_folder = "all_dataset"
    output_folder = "all_rotate_dataset"
    crop_ratio = 0.5

    process_images(input_folder, output_folder, crop_ratio)
