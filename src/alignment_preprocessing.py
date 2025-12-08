import os
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision.transforms.functional import to_pil_image

mtcnn = MTCNN(image_size=224, margin=20)

def align_folder(input_root, output_root):
    print(f"[ALIGN] Reading: {input_root}")

    os.makedirs(output_root, exist_ok=True)

    for person in os.listdir(input_root):
        person_input = os.path.join(input_root, person)
        person_output = os.path.join(output_root, person)

        if not os.path.isdir(person_input):
            continue

        os.makedirs(person_output, exist_ok=True)

        for img_name in os.listdir(person_input):
            if not img_name.lower().endswith(("jpg", "jpeg", "png")):
                continue

            in_path = os.path.join(person_input, img_name)
            out_path = os.path.join(person_output, img_name)

            img = Image.open(in_path).convert("RGB")

            aligned = mtcnn(img)

            if aligned is None:
                print(f"[WARNING] No face detected in {img_name}")
                continue

            # aligned is a Tensor → convert to PIL
            aligned_pil = to_pil_image(aligned)

            aligned_pil.save(out_path)

if __name__ == "__main__":
    align_folder("data/resized_lfw", "data/aligned_lfw")
    align_folder("data/resized_vgg/train", "data/aligned_vgg/train")
    align_folder("data/resized_vgg/test", "data/aligned_vgg/test")
