import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
import os
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess

# ================= CONFIG =================

CELEBRITY_CLASSES = [
    "Angelina Jolie",
    "Brad Pitt",
    "Jennifer Lawrence",
    "Johnny Depp",
    "Leonardo DiCaprio",
    "Natalie Portman",
    "Scarlett Johansson",
    "Tom Cruise",
    "Will Smith",
    "Tom Hanks"
]

MODEL_PATHS = {
    "ResNet50": "resnet50_best.h5",
    "EfficientNet": "efficientnet_best.h5",
    "VGG16": "vgg16_model.h5"
}

IMAGE_SIZE = (224, 224)
CONF_THRESHOLD = 60  # %

# ================= MODEL HANDLER =================

class ModelHandler:
    def __init__(self):
        self.models = {}
        self.active_model = None
        self.active_model_name = None

    def load_model(self, name):
        """Load a model by name, caching it for future use"""
        if name in self.models:
            self.active_model = self.models[name]
            self.active_model_name = name
            return True

        path = MODEL_PATHS.get(name)
        if not path or not os.path.exists(path):
            return False

        try:
            model = load_model(path)
            self.models[name] = model
            self.active_model = model
            self.active_model_name = name
            return True
        except Exception as e:
            print(f"Error loading model {name}: {e}")
            return False

    def preprocess_image(self, img):
        """Preprocess image according to the active model's requirements"""
        if isinstance(img, Image.Image):
            img = np.array(img)

        img = cv2.resize(img, IMAGE_SIZE)

        # Convert grayscale to RGB if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)

        # Apply model-specific preprocessing
        if self.active_model_name == "ResNet50":
            img = resnet_preprocess(img)
        elif self.active_model_name == "EfficientNet":
            img = efficientnet_preprocess(img)
        elif self.active_model_name == "VGG16":
            img = vgg_preprocess(img)

        return img

    def predict(self, img):
        """Predict celebrity from image"""
        if self.active_model is None:
            return []

        x = self.preprocess_image(img)
        preds = self.active_model.predict(x, verbose=0)
        
        # Handle different output formats
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        
        preds = preds[0] if len(preds.shape) > 1 else preds

        # Get top 3 predictions
        top_idx = np.argsort(preds)[-3:][::-1]
        results = []

        for i in top_idx:
            results.append({
                "name": CELEBRITY_CLASSES[i],
                "confidence": float(preds[i] * 100),
                "class_idx": i
            })

        # Return "Unknown" if confidence is too low
        if results[0]["confidence"] < CONF_THRESHOLD:
            return [{"name": "Unknown", "confidence": results[0]["confidence"], "class_idx": None}]

        return results

    def generate_gradcam(self, img, class_idx):
        """Generate Grad-CAM heatmap for the predicted class"""
        if class_idx is None:
            return np.array(img)
        
        model = self.active_model

        # Find the last convolutional layer
        last_conv_layer_name = None
        
        if self.active_model_name == "EfficientNet":
            last_conv_layer_name = "top_conv"
        elif self.active_model_name == "ResNet50":
            last_conv_layer_name = "conv5_block3_out"
        elif self.active_model_name == "VGG16":
            last_conv_layer_name = "block5_conv3"
        
        # Fallback: find last conv layer automatically
        if last_conv_layer_name is None or last_conv_layer_name not in [l.name for l in model.layers]:
            for layer in reversed(model.layers):
                if len(layer.output_shape) == 4:
                    last_conv_layer_name = layer.name
                    break
        
        if last_conv_layer_name is None:
            print("No convolutional layer found")
            return np.array(img)

        try:
            # Create Grad-CAM model
            grad_model = tf.keras.models.Model(
                inputs=model.inputs,
                outputs=[
                    model.get_layer(last_conv_layer_name).output,
                    model.output
                ]
            )

            x = self.preprocess_image(img)

            # Compute gradient
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(x)
                
                # Handle different output formats
                if isinstance(predictions, (list, tuple)):
                    predictions = predictions[0]
                
                loss = predictions[:, class_idx]

            # Calculate gradients
            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            # Generate heatmap
            conv_outputs = conv_outputs[0]
            heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

            # Normalize heatmap
            heatmap = tf.maximum(heatmap, 0)
            heatmap /= (tf.reduce_max(heatmap) + 1e-8)
            heatmap = heatmap.numpy()

            # Resize heatmap to match image size
            heatmap = cv2.resize(heatmap, IMAGE_SIZE)

            # Convert to RGB
            img_array = cv2.resize(np.array(img), IMAGE_SIZE)
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

            # Apply colormap
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # Superimpose heatmap on original image
            superimposed = cv2.addWeighted(img_array, 0.6, heatmap, 0.4, 0)
            return superimposed

        except Exception as e:
            print(f"Error generating Grad-CAM: {e}")
            return np.array(img)


# ================= GUI =================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class App:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.geometry("1500x850")
        self.root.title("Celebrity Face Recognition 🎬 (Grad-CAM + Real-Time)")

        self.model_handler = ModelHandler()
        self.current_image = None
        self.cap = None
        self.webcam_active = False

        self.build_ui()
        self.load_first_model()

    def load_first_model(self):
        """Load the first available model"""
        for m in MODEL_PATHS:
            if self.model_handler.load_model(m):
                self.model_menu.set(m)
                self.status.configure(text=f"✅ {m} Ready", text_color="#22c55e")
                return
        self.status.configure(text="❌ No model found", text_color="red")

    def build_ui(self):
        """Build the user interface"""
        # Header
        header = ctk.CTkLabel(
            self.root, 
            text="🎬 Celebrity Face Recognition",
            font=ctk.CTkFont(size=30, weight="bold")
        )
        header.pack(pady=10)

        # Top controls
        top = ctk.CTkFrame(self.root)
        top.pack()

        self.model_menu = ctk.CTkOptionMenu(
            top, 
            values=list(MODEL_PATHS.keys()), 
            command=self.change_model, 
            width=220
        )
        self.model_menu.pack(side="left", padx=10)

        self.status = ctk.CTkLabel(top, text="Loading...")
        self.status.pack(side="left", padx=20)

        # Main body with two panels
        body = ctk.CTkFrame(self.root)
        body.pack(fill="both", expand=True, padx=20, pady=20)

        # Left panel: Original image
        left_panel = ctk.CTkFrame(body)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        ctk.CTkLabel(
            left_panel, 
            text="Original Image", 
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=5)
        
        self.image_label = ctk.CTkLabel(left_panel, text="")
        self.image_label.pack(fill="both", expand=True)

        # Right panel: Grad-CAM + Results
        right_panel = ctk.CTkFrame(body)
        right_panel.pack(side="right", fill="both", expand=True, padx=(10, 0))
        
        ctk.CTkLabel(
            right_panel, 
            text="Grad-CAM Visualization", 
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=5)
        
        self.gradcam_label = ctk.CTkLabel(right_panel, text="")
        self.gradcam_label.pack(fill="both", expand=True)
        
        # Results display
        self.results_label = ctk.CTkLabel(
            right_panel,
            text="",
            font=ctk.CTkFont(size=14),
            justify="left"
        )
        self.results_label.pack(pady=10)

        # Bottom buttons
        bottom = ctk.CTkFrame(self.root)
        bottom.pack(pady=10)

        ctk.CTkButton(
            bottom, 
            text="📁 Upload Image",
            command=self.upload_image, 
            width=200,
            height=40,
            font=ctk.CTkFont(size=14)
        ).pack(side="left", padx=5)

        self.webcam_btn = ctk.CTkButton(
            bottom, 
            text="📹 Start Webcam",
            command=self.toggle_webcam, 
            width=200,
            height=40,
            font=ctk.CTkFont(size=14)
        )
        self.webcam_btn.pack(side="left", padx=5)

    def change_model(self, name):
        """Change the active model"""
        self.status.configure(text=f"Loading {name}...", text_color="yellow")
        if self.model_handler.load_model(name):
            self.status.configure(text=f"✅ {name} Ready", text_color="#22c55e")
            # Reprocess current image if available
            if self.current_image:
                self.process_frame(self.current_image)
        else:
            self.status.configure(text=f"❌ {name} not found", text_color="red")

    def upload_image(self):
        """Upload and process an image"""
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.png *.jpeg *.bmp *.gif")]
        )
        if path:
            img = Image.open(path).convert('RGB')
            self.current_image = img
            self.process_frame(img)

    def show_image(self, label, img):
        """Display an image in a label"""
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        
        img = img.copy()
        img.thumbnail((650, 650))
        photo = ImageTk.PhotoImage(img)
        label.configure(image=photo, text="")
        label.image = photo

    def process_frame(self, img):
        """Process a frame and display results"""
        # Get predictions
        preds = self.model_handler.predict(img)
        
        if not preds:
            self.results_label.configure(text="No predictions available")
            return

        # Display original image
        self.show_image(self.image_label, img)

        # Generate and display Grad-CAM if prediction is confident
        if preds[0]["name"] != "Unknown" and preds[0].get("class_idx") is not None:
            cam = self.model_handler.generate_gradcam(img, preds[0]["class_idx"])
            self.show_image(self.gradcam_label, cam)
        else:
            # Show original image if prediction is unknown
            self.show_image(self.gradcam_label, img)

        # Display results text
        results_text = "Predictions:\n\n"
        for i, p in enumerate(preds, 1):
            results_text += f"#{i} {p['name']}\n"
            results_text += f"   Confidence: {p['confidence']:.1f}%\n\n"
        
        self.results_label.configure(text=results_text)

    def toggle_webcam(self):
        """Toggle webcam on/off"""
        if not self.webcam_active:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.status.configure(text="❌ Cannot open webcam", text_color="red")
                return
            
            self.webcam_active = True
            self.webcam_btn.configure(text="⏹ Stop Webcam", fg_color="red")
            threading.Thread(target=self.webcam_loop, daemon=True).start()
        else:
            self.webcam_active = False
            if self.cap:
                self.cap.release()
            self.webcam_btn.configure(text="📹 Start Webcam", fg_color=["#3B8ED0", "#1F6AA5"])

    def webcam_loop(self):
        """Process webcam frames in real-time"""
        while self.webcam_active:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                self.process_frame(img)
            else:
                break

    def run(self):
        """Start the application"""
        self.root.mainloop()


# ================= RUN =================

if __name__ == "__main__":
    app = App()
    app.run()
