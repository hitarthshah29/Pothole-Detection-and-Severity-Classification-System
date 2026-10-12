"""
prepare_data.py

This script prepares the raw pothole dataset for an experimental image-classification model.
It extracts proxy severity labels based on the relative bounding box area of potholes in the image.

IMPORTANT LIMITATION:
These classes are generated from bounding-box area and must be described as proxy severity labels.
They do not represent verified real-world pothole severity. This method does not measure depth, 
structural damage, or road safety.

Limitation on overlapping boxes:
For simplicity, if an image has multiple potholes, we simply sum their bounding box areas.
If the boxes overlap, the area might be slightly overestimated, but we cap the total relative area at 1.0.
"""

import os
import json
import shutil
import xml.etree.ElementTree as ET

# Configuration paths (relative to the pothole_detection directory)
DATASET_DIR = "archive/annotated-images"
SPLITS_FILE = "archive/splits.json"
OUTPUT_DIR = "data/processed"
REPORT_FILE = "outputs/preparation_report.txt"

# Provisional Thresholds for Proxy Severity
# These are experimental thresholds and can be adjusted later.
# They are not medically or engineering-validated severity standards.
THRESHOLD_MINOR = 0.05
THRESHOLD_SEVERE = 0.15

def get_severity_label(relative_area):
    """
    Assigns an experimental proxy severity label based on the relative area.
    """
    if relative_area < THRESHOLD_MINOR:
        return "Minor"
    elif relative_area < THRESHOLD_SEVERE:
        return "Moderate"
    else:
        return "Severe"

def parse_xml_annotation(xml_path):
    """
    Parses the PASCAL VOC XML file to extract image size and total pothole area.
    Returns relative_area (float) or None if parsing fails.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Get image dimensions
        size_elem = root.find("size")
        if size_elem is None:
            return None
        
        width_elem = size_elem.find("width")
        height_elem = size_elem.find("height")
        
        if width_elem is None or height_elem is None:
            return None
            
        width = float(width_elem.text)
        height = float(height_elem.text)
        image_area = width * height
        
        if image_area <= 0:
            return None

        # Calculate total pothole area
        total_pothole_area = 0.0
        for obj in root.findall("object"):
            name = obj.find("name")
            if name is not None and name.text.lower() == "pothole":
                bndbox = obj.find("bndbox")
                if bndbox is not None:
                    xmin = float(bndbox.find("xmin").text)
                    ymin = float(bndbox.find("ymin").text)
                    xmax = float(bndbox.find("xmax").text)
                    ymax = float(bndbox.find("ymax").text)
                    
                    # Calculate box area
                    box_width = xmax - xmin
                    box_height = ymax - ymin
                    
                    if box_width > 0 and box_height > 0:
                        total_pothole_area += (box_width * box_height)
        
        relative_area = total_pothole_area / image_area
        
        # Cap at 1.0 in case of overlapping boxes
        return min(relative_area, 1.0)
        
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return None

def create_directory_structure():
    """
    Creates the processed data folder structure with train/test and severity subfolders.
    """
    splits = ["train", "test"]
    categories = ["Minor", "Moderate", "Severe"]
    
    for split in splits:
        for category in categories:
            folder_path = os.path.join(OUTPUT_DIR, split, category)
            os.makedirs(folder_path, exist_ok=True)

def main():
    print("Starting dataset preparation...")
    
    # Check if necessary paths exist
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory '{DATASET_DIR}' not found.")
        return
        
    if not os.path.exists(SPLITS_FILE):
        print(f"Error: Splits file '{SPLITS_FILE}' not found.")
        return

    # Load splits JSON
    try:
        with open(SPLITS_FILE, 'r') as f:
            splits_data = json.load(f)
    except Exception as e:
        print(f"Error loading {SPLITS_FILE}: {e}")
        return

    # Validate splits structure
    if "train" not in splits_data or "test" not in splits_data:
        print("Error: splits.json does not contain 'train' and 'test' keys. Stopping.")
        return

    # Create destination folders
    create_directory_structure()

    stats = {
        "processed": 0,
        "skipped": 0,
        "missing_image": 0,
        "malformed_xml": 0,
        "train": {"Minor": 0, "Moderate": 0, "Severe": 0},
        "test": {"Minor": 0, "Moderate": 0, "Severe": 0}
    }

    # Process each split (train and test)
    for split_name in ["train", "test"]:
        xml_list = splits_data[split_name]
        print(f"Processing {split_name} split ({len(xml_list)} files)...")
        
        for xml_filename in xml_list:
            xml_path = os.path.join(DATASET_DIR, xml_filename)
            
            # The image filename is usually the same as XML but with .jpg
            base_name = os.path.splitext(xml_filename)[0]
            img_filename = base_name + ".jpg"
            img_path = os.path.join(DATASET_DIR, img_filename)
            
            # Check for file existence
            if not os.path.exists(xml_path):
                print(f"Warning: XML missing - {xml_filename}")
                stats["skipped"] += 1
                stats["malformed_xml"] += 1
                continue
                
            if not os.path.exists(img_path):
                print(f"Warning: Image missing - {img_filename}")
                stats["skipped"] += 1
                stats["missing_image"] += 1
                continue
            
            # Calculate severity proxy using XML bounding box data
            relative_area = parse_xml_annotation(xml_path)
            
            if relative_area is None:
                print(f"Warning: Failed to parse or find valid area in {xml_filename}")
                stats["skipped"] += 1
                stats["malformed_xml"] += 1
                continue
                
            severity = get_severity_label(relative_area)
            
            # Copy original image to appropriate category folder
            dest_folder = os.path.join(OUTPUT_DIR, split_name, severity)
            dest_path = os.path.join(dest_folder, img_filename)
            
            try:
                shutil.copy2(img_path, dest_path)
                stats["processed"] += 1
                stats[split_name][severity] += 1
            except Exception as e:
                print(f"Error copying {img_filename}: {e}")
                stats["skipped"] += 1

    # Write final statistics report
    try:
        os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
        with open(REPORT_FILE, 'w') as f:
            f.write("Dataset Preparation Report\n")
            f.write("==========================\n\n")
            f.write("Methodology:\n")
            f.write("- Labels are PROXY severities derived mathematically from relative bounding box area.\n")
            f.write("- These DO NOT represent engineering-verified real-world severity (depth, damage, etc.).\n")
            f.write("- If multiple boxes exist in an image, their areas are summed (capped at 1.0).\n\n")
            f.write(f"Thresholds Used:\n")
            f.write(f"- Minor: relative area < {THRESHOLD_MINOR}\n")
            f.write(f"- Moderate: {THRESHOLD_MINOR} <= relative area < {THRESHOLD_SEVERE}\n")
            f.write(f"- Severe: relative area >= {THRESHOLD_SEVERE}\n\n")
            f.write("Data Split Details:\n")
            f.write("- Split method: Predefined split from archive/splits.json was used.\n\n")
            f.write("Processing Statistics:\n")
            f.write(f"- Successfully processed files: {stats['processed']}\n")
            f.write(f"- Total skipped files: {stats['skipped']}\n")
            f.write(f"  - Due to missing images: {stats['missing_image']}\n")
            f.write(f"  - Due to malformed/missing XML: {stats['malformed_xml']}\n\n")
            f.write("Class Distribution (Train):\n")
            f.write(f"- Minor: {stats['train']['Minor']}\n")
            f.write(f"- Moderate: {stats['train']['Moderate']}\n")
            f.write(f"- Severe: {stats['train']['Severe']}\n\n")
            f.write("Class Distribution (Test):\n")
            f.write(f"- Minor: {stats['test']['Minor']}\n")
            f.write(f"- Moderate: {stats['test']['Moderate']}\n")
            f.write(f"- Severe: {stats['test']['Severe']}\n")
        print(f"\nPreparation complete. Report saved to {REPORT_FILE}")
    except Exception as e:
        print(f"Error writing report: {e}")

if __name__ == "__main__":
    main()
