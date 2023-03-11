from PIL import Image
from clip_interrogator import Config, Interrogator
import os
import argparse

if __name__ == '__main__':
    ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_folder","-r", type=str, default=None,
                        help="folder name")
    args = parser.parse_args()
    images_folder=args.images_folder
    if images_folder is None:
        print("Folder name is needed")
        exit()
    for filename in os.listdir(images_folder):
        image_path = os.path.join(images_folder, filename)
        if os.path.isfile(image_path):
            image = Image.open(image_path).convert('RGB')
            prompt=ci.interrogate(image)
            base_name, extension = os.path.splitext(filename)
            prompt_filename = base_name + ".txt"
            prompt_filepath = os.path.join(images_folder, prompt_filename)
            with open(prompt_filepath, 'w') as file:
                file.write(prompt)