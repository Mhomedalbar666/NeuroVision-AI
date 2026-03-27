import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import gradio as gr
import numpy as np

# ===============================
# Model definition
# ===============================
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
    def forward(self, x):
        return self.block(x)

class Model_v2(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512)
        )
        self.cam_conv = nn.Conv2d(512, num_classes, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        x = self.features(x)
        heatmap = self.cam_conv(x)
        logits = self.pool(heatmap).view(x.size(0), -1)
        return logits, heatmap

# ===============================
# Load models
# ===============================
device = "cuda" if torch.cuda.is_available() else "cpu"

model_ct_stroke = Model_v2(1).to(device)
model_ct_stroke.load_state_dict(torch.load("modelv2_weights_another.pth", map_location=device))
model_ct_stroke.eval()

model_mr_stroke = Model_v2(1).to(device)
model_mr_stroke.load_state_dict(torch.load("modelv1_mr_weights.pth", map_location=device))
model_mr_stroke.eval()

model_ct_tumor = Model_v2(1).to(device)
model_ct_tumor.load_state_dict(torch.load("model_tumor_weights.pth", map_location=device))
model_ct_tumor.eval()

model_mr_tumor = Model_v2(1).to(device)
model_mr_tumor.load_state_dict(torch.load("model_weights_tumor_mri_second.pth", map_location=device))
model_mr_tumor.eval()

# ===============================
# Transforms
# ===============================
ct_transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.clamp(x,-1000,400)/1400)
])
mri_transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x-x.mean())/(x.std()+1e-5))
])

# ===============================
# Utility
# ===============================
def preprocess_image(img, modality):
    if img is None:
        return None
    if img.mode != "L":
        img = img.convert("L")
    tensor = ct_transform(img).unsqueeze(0).to(device) if modality=="CT" else mri_transform(img).unsqueeze(0).to(device)# we used unsqueeze()->to git rid from the batches because they are in index zero
    
    return tensor

def generate_heatmap(model, input_tensor, original_image, color_map="inferno"):
    with torch.inference_mode():
        y_pred, heatmap = model(input_tensor)
        y_pred_soft = torch.softmax(y_pred, dim=1)[0].cpu().numpy()
    predicted = int(np.argmax(y_pred_soft))
    cam = heatmap[0, predicted].detach().cpu().numpy()#because just we want the height and width
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    import matplotlib.cm as cm
    cam_color = cm.get_cmap(color_map)(cam)[:, :, :3]
    cam_color = (cam_color*255).astype(np.uint8)
    cam_color = Image.fromarray(cam_color).resize(original_image.size, Image.BILINEAR)#this make th width and height correct to each other beacuse in pytorch [h,w] but in PIL [w,h]
    overlay_img = Image.blend(original_image.convert("RGBA"), cam_color.convert("RGBA"), alpha=0.5)
    return overlay_img, y_pred_soft, predicted

CLASS_NAMES = ["Normal","Stroke"]
TUMOR_CLASS_NAMES = ["Normal","Tumor"]
def run_models(img, modality):
    input_tensor = preprocess_image(img, modality)
    if input_tensor is None:
        return None
    models = {"Tumor": model_ct_tumor, "Stroke": model_ct_stroke} if modality=="CT" else {"Tumor": model_mr_tumor, "Stroke": model_mr_stroke}
    results = {}
    for name, model in models.items():
        cmap = "plasma" if name=="Tumor" else "inferno"
        overlay, probs, pred_idx = generate_heatmap(model, input_tensor, img, color_map=cmap)
        pred_text = "Tumor" if (CLASS_NAMES[pred_idx]=="Stroke" and name=="Tumor") else CLASS_NAMES[pred_idx]
        results[name] = {"overlay": overlay, "probs": probs, "prediction": pred_text}
    return results

# ===============================
# Treatment advice
# ===============================
def treatment_advice_interface(results, age_category, language):
    advice = []

    # اختر اللغة أولاً
    language_text = f"<div style='font-size:16px; font-weight:bold;'>Language: {'Arabic' if language=='AR' else 'English'}</div>"
    advice.append(language_text)

    combined_prediction = [v["prediction"] for v in results.values()]
    has_tumor = "Tumor" in combined_prediction
    has_stroke = "Stroke" in combined_prediction
    high_risk = has_tumor or has_stroke

    if high_risk:
        advice.append("<div style='color:red; font-weight:bold; font-size:18px;'>⚠️ High Risk Detected!</div>")

    # نصائح حسب الحالة
    if has_tumor and has_stroke:
        if language=="AR":
            advice.extend([
                "🚨 حالة حرجة جدًا: وجود ورم دماغي مع سكتة دماغية.",
                "⚠️ يتطلب إدخال فوري للمستشفى والتعامل مع فريق طبي متعدد التخصصات.",
                "🧠 تدخل جراحي + عناية مركزة + تصوير عصبي عاجل (MRI / CT Perfusion).",
                "📌 المتابعة مع: جراحة الأعصاب، الأعصاب، والأورام العصبية."
            ])
        else:
            advice.extend([
                "🚨 Critical condition: Brain tumor combined with stroke detected.",
                "⚠️ Immediate hospital admission required with multidisciplinary team.",
                "🧠 Surgical evaluation + ICU care + urgent neuro-imaging (MRI / CT Perfusion).",
                "📌 Follow-up with neurosurgery, neurology, and neuro-oncology."
            ])
    elif has_tumor:
        if language=="AR":
            advice.extend([
                "🧠 تم اكتشاف ورم دماغي.",
                "🔬 يلزم تحديد نوع الورم (حميد / خبيث) باستخدام MRI مع تباين."
            ])
            if age_category=="Child":
                advice.append("👶 متابعة مع طبيب أورام أعصاب أطفال.")
            elif age_category=="Adult":
                advice.append("🧑 استشارة جراحة أعصاب وأورام.")
            else:
                advice.append("👴 مراعاة الحالة العامة والعلاج المحافظ إن لزم.")
            advice.append("⚕️ خيارات العلاج: جراحة، إشعاع، علاج كيميائي حسب الحالة.")
        else:
            advice.extend([
                "🧠 Brain tumor detected.",
                "🔬 Tumor typing required (benign vs malignant) via contrast MRI.",
                "⚕️ Treatment options: surgery, radiotherapy, chemotherapy."
            ])
    elif has_stroke:
        if language=="AR":
            advice.extend([
                "⚠️ سكتة دماغية محتملة.",
                "🚑 حالة طبية طارئة – الوقت عامل حاسم.",
                "🩺 تصوير CT / MRI عاجل لتحديد النوع (إقفارية / نزفية).",
                "💊 علاج فوري لمنع تلف الدماغ الدائم."
            ])
        else:
            advice.extend([
                "⚠️ Stroke suspected.",
                "🚑 Medical emergency – time-critical condition.",
                "🩺 Urgent CT/MRI to determine ischemic or hemorrhagic stroke.",
                "💊 Immediate intervention to prevent permanent damage."
            ])
    else:
        if language=="AR":
            advice.extend([
                "✅ لا توجد مؤشرات مرضية واضحة.",
                "🧘‍♂️ الاستمرار بنمط حياة صحي والمتابعة الدورية."
            ])
        else:
            advice.extend([
                "✅ No abnormal findings detected.",
                "🧘‍♂️ Maintain healthy lifestyle and routine checkups."
            ])

    # مصادر موثوقة
    if language=="AR":
        advice.extend([
            "🔗 مصادر طبية موثوقة:",
            "• منظمة الصحة العالمية: https://www.who.int",
            "• مايو كلينك: https://www.mayoclinic.org",
            "• NIH: https://www.nih.gov"
        ])
    else:
        advice.extend([
            "🔗 Trusted Medical Resources:",
            "• World Health Organization: https://www.who.int",
            "• Mayo Clinic: https://www.mayoclinic.org",
            "• National Institutes of Health: https://www.nih.gov"
        ])

    return "<div style='font-size:16px; font-weight:bold; line-height:1.5; color:#1E90FF;'>"+ "<br>".join(advice) + "</div>"

# ===============================
# Gradio Interface
# ===============================
with gr.Blocks() as iface:
    gr.Markdown("<h1 style='text-align:center; color:#4B0082;'>🧠 Smart Brain Analyzer</h1>")

    with gr.Row():
        with gr.Column():
            modality_select = gr.Radio(["CT","MRI"], label="Image Modality", value="CT")
            input_image = gr.Image(type="pil", label="Upload Brain Image")
            language_select = gr.Radio(["EN","AR"], label="Language", value="EN")  # فوق النصائح
            age_select = gr.Radio(["Child","Adult","Elderly"], label="Age Category", value="Adult")
            treatment_btn = gr.Button("💊 Treatment Advice")
            clear_btn = gr.Button("♻️ Clear & Reset")
            advice_box = gr.Markdown("Advice will appear here...", elem_id="advice_box")

        with gr.Column():
            # عنوان فوق Heatmap
            
            tumor_heatmap = gr.Image(label="", type="pil")
            tumor_prob = gr.Label(num_top_classes=2, label="tumor probabilities")  # الاحتمال تحت Heatmap مباشرة

            gr.Markdown("### Stroke")
            stroke_heatmap = gr.Image(label="", type="pil")
            stroke_prob = gr.Label(num_top_classes=2, label="stroke probabilities")
            gr.Markdown("<hr><p style='text-align:center; font-size:18px; font-weight:bold; color:#FF4500;'>Made by Soft Engineer Mohammed AL-Bar</p>")

    # ===============================
    # Callbacks
    # ===============================
    
    
    def predict_all(img, modality):
        results = run_models(img, modality)
        if results is None:
            return None, None, {}, {}

        return (
        results["Tumor"]["overlay"],
        results["Stroke"]["overlay"],
        {TUMOR_CLASS_NAMES[i]: float(results["Tumor"]["probs"][i]) for i in range(2)},
        {CLASS_NAMES[i]: float(results["Stroke"]["probs"][i]) for i in range(2)}
    )
    def advice_all(img, modality, age_cat, language):
        results = run_models(img, modality)
        if results is None:
            return "<div style='color:gray;'>No image uploaded.</div>"
        return treatment_advice_interface(results, age_cat, language)

    def clear_all():
        return None, None, {}, {}, "<div style='color:gray;'>Advice will appear here...</div>"

    input_image.change(fn=predict_all, inputs=[input_image, modality_select],
                       outputs=[tumor_heatmap, stroke_heatmap, tumor_prob, stroke_prob])
    treatment_btn.click(fn=advice_all, inputs=[input_image, modality_select, age_select, language_select],
                        outputs=advice_box)
    clear_btn.click(fn=clear_all, inputs=[], outputs=[tumor_heatmap, stroke_heatmap, tumor_prob, stroke_prob, advice_box])

iface.launch(theme=gr.themes.Soft(), share=False)
