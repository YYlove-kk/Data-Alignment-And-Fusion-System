import os
import argparse
import glob
import numpy as np
import cv2
import pydicom
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch import nn
from util.time2vec import Time2Vec
from datetime import datetime
import json
import sys

def load_dicom_images(folder_path):
    dicom_info = []
    dcm_files = glob.glob(os.path.join(folder_path, "**", "*.dcm"), recursive=True)
    for path in dcm_files:
        try:
            dicom_data = pydicom.dcmread(path)
            if hasattr(dicom_data, "ImagePositionPatient"):
                z = dicom_data.ImagePositionPatient[2]
                dicom_info.append((z, dicom_data.pixel_array))
        except Exception as e:
            print(f"❌ Error reading {path}: {e}", file=sys.stderr)
    dicom_info.sort(key=lambda x: x[0])
    return [img for _, img in dicom_info]


def extract_acquisition_date(folder_path):
    dates = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".dcm"):
                path = os.path.join(root, file)
                try:
                    dicom = pydicom.dcmread(path)
                    if hasattr(dicom, "AcquisitionDate"):
                        dates.append(dicom.AcquisitionDate)
                except Exception as e:
                    print(f"⚠️ Failed to read DICOM date from {path}: {e}", file=sys.stderr)
    if dates:
        return min(dates)
    return "20000101"


def denoise_image(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


def normalize_image(image):
    return image.astype(np.float32) / np.max(image) if np.max(image) > 0 else image


def resize_image(image, target_size=(224, 224)):
    return cv2.resize(image, target_size)


def convert_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def build_resnet_model():
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    modules = list(resnet.children())[:-1]
    model = nn.Sequential(*modules)
    model.eval()
    return model


def extract_features(image, model):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = model(input_tensor)
    return features.squeeze().cpu()


def process_source_images(source_folder, model, time_encoder):
    dicom_images = load_dicom_images(source_folder)
    acquisition_date = extract_acquisition_date(source_folder)
    acquisition_date_dt = datetime.strptime(acquisition_date, "%Y%m%d")
    time_vector = time_encoder(acquisition_date_dt)

    feature_vectors = []
    for image in dicom_images:
        try:
            image = denoise_image(image)
            image = normalize_image(image)
            image = resize_image(image)
            image = convert_to_rgb(image)
            image = image.astype(np.float32)
            feature_vector = extract_features(image, model)
            combined = torch.cat((feature_vector, time_vector.squeeze(0)), dim=0)
            feature_vectors.append(combined)
        except Exception as e:
            print(f"❌ Error processing image: {e}", file=sys.stderr)

    return torch.stack(feature_vectors) if feature_vectors else None


def save_source_embedding(features, output_dir,patient_name, patient_folder):
    acquisition_date = extract_acquisition_date(patient_folder)
    timestamp = acquisition_date
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{patient_name}_{timestamp}_img.npy"
    path = os.path.join(output_dir, filename)
    np.save(path, features.detach().numpy())
    return filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process DICOM images and extract features.")
    parser.add_argument("--source_folder", required=True, help="Path to the raw DICOM folder")
    parser.add_argument("--output_dir", required=True, help="Path to save the feature embeddings")

    args = parser.parse_args()

    source_folder = args.source_folder
    output_dir = args.output_dir
    source_id = os.path.basename(source_folder.rstrip("/"))

    model = build_resnet_model()
    time_encoder = Time2Vec(32)

    result_paths = []

    for patient_name in sorted(os.listdir(source_folder)):
        patient_path = os.path.join(source_folder, patient_name)
        if not os.path.isdir(patient_path):
            continue

        print(f"🔍 Processing patient: {patient_name}", file=sys.stderr)
        features = process_source_images(patient_path, model, time_encoder)
        if features is not None:
            save_path = save_source_embedding(features, output_dir,patient_name, patient_path)
            result_paths.append(save_path)
            print(f"✅ Saved for {patient_name} -> {save_path}", file=sys.stderr)

    # ✅ 只输出路径 JSON 给 Java 读取
    print(json.dumps({"paths": result_paths}))
