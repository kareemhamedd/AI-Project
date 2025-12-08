import os
from PIL import Image
import torchvision.transforms as transforms

def preprocess_image(img_path, output_path):
    img = Image.open(img_path).convert("RGB")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224))
    ])

    img = transform(img)
    img.save(output_path)

def preprocess_folder(input_root, output_root):
    print(f"[RESIZE] Reading: {input_root}")
    
    os.makedirs(output_root, exist_ok=True)

    for person in os.listdir(input_root):
        person_input = os.path.join(input_root, person)
        person_output = os.path.join(output_root, person)

        if not os.path.isdir(person_input):
            continue  

        os.makedirs(person_output, exist_ok=True)

        for img_name in os.listdir(person_input):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(person_input, img_name)
            out_path = os.path.join(person_output, img_name)
            preprocess_image(img_path, out_path)

if __name__ == "__main__":
    # LFW
    preprocess_folder("data/raw/lfw/lfw-deepfunneled/lfw-deepfunneled", 
                      "data/resized_lfw")

    # VGGFace2 Mini
    preprocess_folder("data/raw/vggface2/mini_train", "data/resized_vgg/train")
    preprocess_folder("data/raw/vggface2/mini_test", "data/resized_vgg/test")
