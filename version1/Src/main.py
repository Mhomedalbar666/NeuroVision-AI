# app_super_overlay_animated_final.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import gradio as gr
import PIL.Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import io

# Conv Block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
    def forward(self, x):
        return self.block(x)

# Full MRI Model
class MRIModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load full model
MODEL_PATH = "mri_model_full (1).pth"
model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.to(device)
model.eval()

# Class names
CLASS_NAMES = ["Tumor", "Normal"]

# Image Transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def preprocess_image(img: PIL.Image.Image):
    if img is None:
        return None
    if img.mode != "RGB":
        img = img.convert("RGB")
    tensor = transform(img).unsqueeze(0)
    return tensor.to(device)


# Heatmap + Overlay Animated
def generate_animated_overlay(input_tensor, original_image, frames=5):
    features = None
    def hook_fn(module, inp, outp):
        nonlocal features
        features = outp
    handle = model.features[-1].register_forward_hook(hook_fn)

    with torch.inference_mode():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    handle.remove()

    predicted_idx = np.argmax(probs)

    fmap = features[0].mean(dim=0).cpu().numpy()
    fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)

    overlay_frames = []
    for i in range(frames):
        alpha_overlay = 0.3 + 0.4*probs[predicted_idx] + 0.1*np.sin(i/frames*2*np.pi)
        cmap = cm.inferno(fmap)[:,:,:3]
        cmap = (cmap*255).astype(np.uint8)
        heatmap_img = PIL.Image.fromarray(cmap).resize(original_image.size)
        overlay_img = PIL.Image.blend(original_image.convert("RGBA"), heatmap_img.convert("RGBA"), alpha=alpha_overlay)
        overlay_frames.append(overlay_img)

    return overlay_frames[-1], probs, predicted_idx

# Prediction + Probability chart
def predict_image(img):
    input_tensor = preprocess_image(img)
    if input_tensor is None:
        return None, None, None

    overlay_img, probs, predicted_idx = generate_animated_overlay(input_tensor, img)
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = probs[predicted_idx]

    plt.figure(figsize=(5,4))
    cmap = plt.get_cmap("plasma")
    colors = [cmap(p) for p in probs]
    bars = plt.bar(CLASS_NAMES, probs, color=colors)
    plt.ylim(0,1)
    plt.title("Classification Probabilities", fontsize=16, fontweight='bold')
    for bar, p in zip(bars, probs):
        plt.text(bar.get_x()+bar.get_width()/2, p+0.02, f"{p:.2f}", ha='center', fontsize=12)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', transparent=True)
    buf.seek(0)
    plt.close()
    prob_chart = PIL.Image.open(buf)

    return overlay_img, prob_chart, f"{predicted_class} ({confidence*100:.1f}%)"

# Treatment advice
def treatment_advice(prediction_text, age_category):
    advice = ""
    if not prediction_text:
        return "⚠️ Prediction not made yet."
    if "Tumor" in prediction_text:
        if age_category=="Child":
            advice += "👶 Child needs careful follow-up with pediatric neuro-oncologist.\n"
        elif age_category=="Adult":
            advice += "🧑 Review with adult neuro-oncologist and discuss treatment options.\n"
        else:
            advice += "👴 Elderly patient: close monitoring and advanced treatment options.\n"
        advice += "- ⚕️ Surgery, radiotherapy, or chemotherapy depending on tumor.\n"
        advice += "- 🥗 Healthy diet and regular follow-up."
    else:
        advice = "✅ No tumor detected. Continue regular checkups and healthy lifestyle."
    # Make advice text visually large and attractive with markdown
    return f"<div style='font-size:20px; font-weight:bold; color:#1E90FF;'>{advice}</div>"

# Gradio interface
with gr.Blocks() as iface:
    gr.Markdown("<h1 style='text-align:center; color:#4B0082;'>🧠 Smart Brain MRI Project</h1>", elem_id="page_title")
    gr.Markdown("Upload an MRI image to see prediction, animated overlay heatmap, probability chart, and treatment advice!")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload MRI Image")
            treatment_btn = gr.Button("💊 Treatment Advice")
            age_select = gr.Radio(["Child","Adult","Elderly"], label="Age Category", value="Adult")
            clear_btn = gr.Button("♻️ Clear & Reset")
            advice_box = gr.Markdown("Advice will appear here...", elem_id="advice_box")

        with gr.Column():
            overlay_output = gr.Image(label="Overlay Heatmap")
            prob_chart = gr.Image(label="Probability Chart")
            prediction_label = gr.Label(label="Prediction")

    def update_advice(overlay_img, chart_img, pred_text, age_cat):
        return treatment_advice(pred_text, age_cat)

    def clear_all():
        return None, None, "", None

    input_image.change(fn=predict_image, inputs=input_image, outputs=[overlay_output, prob_chart, prediction_label])
    treatment_btn.click(fn=update_advice, inputs=[overlay_output, prob_chart, prediction_label, age_select], outputs=advice_box)
    clear_btn.click(fn=clear_all, inputs=[], outputs=[overlay_output, prob_chart, prediction_label, advice_box])

    # Footer
    gr.Markdown("<hr><p style='text-align:center; font-size:18px; font-weight:bold; color:#FF4500;'>Made by Soft Engineer Mohammed AL-Bar</p>")

# Launch
if __name__=="__main__":
    iface.launch(theme=gr.themes.Soft(), share=False)
