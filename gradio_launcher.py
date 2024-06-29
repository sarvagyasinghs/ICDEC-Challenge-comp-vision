import os
import cv2
import json
import tempfile
from pathlib import Path
from ultralytics import YOLO
from gradio_client import Client, handle_file
import glob
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Load YOLO and OCR models
vehicle_model = YOLO("yolov10s.pt")
license_model = YOLO("/Users/sarvagyasamridhsingh/Documents/Programming/ICDEC Challenge comp vision/juvdv2-vdvwc/licence_weights.pt")
client = Client("gokaygokay/Florence-2")

# Function to detect license plates and run OCR
def detect_and_run_ocr(image):
    results = license_model.predict(
        source=image,
        save=False,
        conf=0.2,
        device='cpu'  # Use CPU
    )
    ocr_results = []
    for r in results:
        if hasattr(r.boxes, 'xyxy') and len(r.boxes.xyxy) > 0:
            for coordinates in r.boxes.xyxy:
                x1, y1, x2, y2 = map(int, coordinates)
                cropped_image = image[y1:y2, x1:x2]
                temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                temp_file_path = temp_file.name
                cv2.imwrite(temp_file_path, cropped_image)
                temp_file.close()
                result = client.predict(
                    image=handle_file(temp_file_path),
                    task_prompt="OCR",
                    text_input=None,
                    model_id="microsoft/Florence-2-large",
                    api_name="/process_image"
                )
                Path(temp_file_path).unlink()
                ocr_result_dict = json.loads(result[0].replace("'", '"'))
                ocr_text = ocr_result_dict.get('<OCR>', '')
                ocr_results.append({
                    'bbox': (x1, y1, x2, y2),
                    'text': ocr_text
                })
                font_scale = 1
                thickness = 2
                text_size, _ = cv2.getTextSize(ocr_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                text_w, text_h = text_size
                cv2.rectangle(image, (x2, y2 - text_h - 10), (x2 + text_w, y2 + 10), (255, 255, 255), cv2.FILLED)
                cv2.putText(image, ocr_text, (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
    return image, ocr_results

# Function to process images from a directory and save results
def process_and_save_images(input_directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    image_paths = glob.glob(os.path.join(input_directory, '**/*.jpg'), recursive=True)
    
    detection_results = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        relative_path = os.path.relpath(image_path, input_directory)
        output_path = os.path.join(output_directory, relative_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Detect vehicles
        vehicle_results = vehicle_model(image, device='cpu')  # Use CPU
        ocr_results = []

        for idx, result in enumerate(vehicle_results):
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            classes = result.boxes.cls.cpu().numpy().astype(int)
            for i, (box, cls) in enumerate(zip(boxes, classes)):
                vehicle_label = f"Vehicle {i+1}"
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(image, vehicle_label, (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                cropped_image = image[box[1]:box[3], box[0]:box[2]]
                annotated_image, ocr_result = detect_and_run_ocr(cropped_image)
                if ocr_result:
                    ocr_results.append({
                        'label': vehicle_label,
                        'bbox': (box[0], box[1], box[2], box[3]),
                        'text': ocr_result[0]['text']
                    })
                else:
                    ocr_results.append({
                        'label': vehicle_label,
                        'bbox': (box[0], box[1], box[2], box[3]),
                        'text': 'No license plate detected'
                    })

        # Save annotated image
        cv2.imwrite(output_path, image)

        # Save OCR results to a text file
        output_txt_path = os.path.splitext(output_path)[0] + '.txt'
        with open(output_txt_path, 'w') as f:
            for result in ocr_results:
                f.write(f"{result['label']} Bounding Box: {result['bbox']}, OCR Text: {result['text']}\n")

        # Collect detection results for mAP calculation
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        for result in ocr_results:
            detection_results.append({
                'image_id': image_id,
                'category_id': 1,  # Assuming one category for vehicles
                'bbox': result['bbox'],
                'score': 1.0  # Assuming all detections are perfect for simplicity
            })

    # Save detection results to JSON for mAP evaluation
    output_json_path = os.path.join(output_directory, 'detection_results.json')
    with open(output_json_path, 'w') as f:
        json.dump(detection_results, f)

# Directories
train_input_dir = "/Users/sarvagyasamridhsingh/Documents/Programming/ICDEC Challenge comp vision/juvdv2-vdvwc/train"
val_input_dir = "/Users/sarvagyasamridhsingh/Documents/Programming/ICDEC Challenge comp vision/juvdv2-vdvwc/val"
train_output_dir = "/Users/sarvagyasamridhsingh/Documents/Programming/ICDEC Challenge comp vision/juvdv2-vdvwc/train_output"
val_output_dir = "/Users/sarvagyasamridhsingh/Documents/Programming/ICDEC Challenge comp vision/juvdv2-vdvwc/val_output"

# Process and save images
process_and_save_images(train_input_dir, train_output_dir)
process_and_save_images(val_input_dir, val_output_dir)
