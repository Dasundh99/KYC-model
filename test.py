from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# Use a smaller model if memory is an issue
model_name = "microsoft/trocr-small-printed"

processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

image = Image.open("id.jpg").convert("RGB")
pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

generated_ids = model.generate(pixel_values)
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("Extracted text:", text)
