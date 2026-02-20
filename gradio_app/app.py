# gradio_app/app.py

import os
import sys

sys.path.append("/content/drive/MyDrive/signature_verification")

import gradio as gr
from PIL import Image
from torchvision import transforms
import torch
from models.encoder import SignatureEncoder

# Load trained model
model = SignatureEncoder()
model.load_state_dict(torch.load("outputs/models/final_model.pt", map_location="cpu"))
model.eval()

# Transform for preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Verification function
def verify_signatures(img1, img2):
    img1 = transform(img1.convert("L")).unsqueeze(0)
    img2 = transform(img2.convert("L")).unsqueeze(0)

    with torch.no_grad():
        emb1 = model(img1)
        emb2 = model(img2)
        distance = torch.nn.functional.pairwise_distance(emb1, emb2, p=2).item()

    # Use trained threshold (adjust based on evaluate.py)
    threshold = 0.7495
    if distance < threshold:
        result = "✔️ Genuine Match"
    else:
        result = "❌ Forged / Mismatch"
    
    return f"{result} | Distance: {distance:.4f} | Threshold: {threshold:.4f}"

# Build interface
demo = gr.Interface(
    fn=verify_signatures,
    inputs=[gr.Image(type="pil", label="Signature 1"), gr.Image(type="pil", label="Signature 2")],
    outputs="text",
    title="Signature Verification (Triplet Model)",
    description="Upload two signature images (one genuine, one test). Uses Triplet Loss & ResNet18 embeddings."
)

# Launch public link
if __name__ == "__main__":
    demo.launch(share=True)
