import os
from PIL import Image
from torchvision import transforms

# Augmentation Pipeline
augment = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Resize((224, 224))
])

def augment_image(img_path, output_path):
    img = Image.open(img_path).convert("RGB")
    img = augment(img)
    img.save(output_path)

def augment_folder(input_root, output_root):
    print(f"[AUGMENT] Reading: {input_root}")

    os.makedirs(output_root, exist_ok=True)

    for person in os.listdir(input_root):
        person_in = os.path.join(input_root, person)
        person_out = os.path.join(output_root, person)

        if not os.path.isdir(person_in):
            continue

        os.makedirs(person_out, exist_ok=True)

        for img_name in os.listdir(person_in):
            if not img_name.lower().endswith(("jpg", "jpeg", "png")):
                continue

            in_path = os.path.join(person_in, img_name)
            out_path = os.path.join(person_out, img_name)

            augment_image(in_path, out_path)

if __name__ == "__main__":
    augment_folder("data/aligned_vgg/train", "data/augmented_vgg/train")
