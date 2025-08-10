import os
from datasets import load_dataset
from PIL import Image

def prepare_flir_dataset():
    print("Downloading FLIR dataset from Hugging Face...")
    dataset = load_dataset("SAE-AAI/FLIR_IR_Expansion")
    print("Download complete.")

    class_names = dataset['train'].features['objects'].feature['category_id'].names
    base_dir = "flir_dataset"
    os.makedirs(os.path.join(base_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "labels", "train"), exist_ok=True)

    print("Processing and converting dataset to YOLOv8 format...")
    for item in dataset['train']:
        image = item['image']
        image_width, image_height = image.size
        image_filename = f"{item['image_id']}.jpg"
        label_filename = f"{item['image_id']}.txt"
        image_path = os.path.join(base_dir, "images", "train", image_filename)
        label_path = os.path.join(base_dir, "labels", "train", label_filename)
        image.save(image_path)

        with open(label_path, 'w') as f:
            for i, bbox in enumerate(item['objects']['bbox']):
                class_id = item['objects']['category_id'][i]
                x_min, y_min, width, height = bbox
                center_x = (x_min + width / 2) / image_width
                center_y = (y_min + height / 2) / image_height
                norm_width = width / image_width
                norm_height = height / image_height
                f.write(f"{class_id} {center_x} {center_y} {norm_width} {norm_height}\n")

    yaml_content = f"""
path: {os.path.abspath(base_dir)}
train: images/train
nc: {len(class_names)}
names: {class_names}
"""
    yaml_path = os.path.join(base_dir, "flir_data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
        
    print(f"Dataset preparation complete. YAML configuration file created at: {yaml_path}")

if __name__ == "__main__":
    prepare_flir_dataset()
