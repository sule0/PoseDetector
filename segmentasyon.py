from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import torch
import numpy as np
# Load the pre-trained model and processor
processor = AutoImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = SegformerForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

# Load and preprocess the image
#image_path = "person.jpg"  
# image = Image.open(requests.get(image_url, stream=True).raw)  # For an image URL
image = Image.open("person3.jpg")  # For a local file

inputs = processor(images=image, return_tensors="pt")

# Run the model
with torch.no_grad():
    outputs = model(**inputs)

# Post-process the output
logits = outputs.logits
segmentation = logits.argmax(dim=1)[0].cpu().numpy()


# Segmentasyon çıktısını normalize edin ve bir renk haritasına dönüştürün
segmentation_normalized = (segmentation - segmentation.min()) / (segmentation.max() - segmentation.min())
segmentation_image = Image.fromarray((segmentation_normalized * 255).astype('uint8'), mode='L')
segmentation_image.show()



