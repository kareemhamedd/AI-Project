import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
import os

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not installed. Install it: pip install tensorflow")

# Configuration
CELEBRITY_CLASSES = [
    "Celebrity 1", "Celebrity 2", "Celebrity 3", "Celebrity 4", "Celebrity 5",
    "Celebrity 6", "Celebrity 7", "Celebrity 8", "Celebrity 9", "Celebrity 10"
]

MODEL_PATHS = {
    "ResNet50": "resnet50_celebrity.h5",
    "VGG16": "vgg16_model.h5",
    "InceptionV3": "inceptionv3_model.h5"
}

IMAGE_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.3

# Model Handler
class ModelHandler:
    def __init__(self):
        self.models = {}
        self.active_model = None
        self.active_model_name = None

    def load_model(self, model_name):
        if model_name in self.models:
            self.active_model = self.models[model_name]
            self.active_model_name = model_name
            return True

        model_path = MODEL_PATHS.get(model_name)
        if not model_path or not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return False

        try:
            if TENSORFLOW_AVAILABLE:
                print(f"Loading {model_name}...")
                model = load_model(model_path)
                self.models[model_name] = model
                self.active_model = model
                self.active_model_name = model_name
                print(f"{model_name} loaded successfully!")
                return True
            else:
                print("TensorFlow not available")
                return False
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            return False

    def preprocess_image(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)

        img = cv2.resize(img, IMAGE_SIZE)

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        return img

    def predict(self, img):
        if self.active_model is None:
            return []

        try:
            processed_img = self.preprocess_image(img)
            predictions = self.active_model.predict(processed_img, verbose=0)[0]
            top_indices = np.argsort(predictions)[-3:][::-1]

            results = []
            for idx in top_indices:
                confidence = float(predictions[idx] * 100)
                if confidence >= CONFIDENCE_THRESHOLD * 100:
                    results.append({
                        "name": CELEBRITY_CLASSES[idx],
                        "confidence": confidence
                    })
            return results
        except Exception as e:
            print(f"Prediction error: {e}")
            return []

    def generate_gradcam(self, img):
        try:
            if self.active_model is None:
                return img

            if isinstance(img, Image.Image):
                img = np.array(img)

            height, width = img.shape[:2]
            y, x = np.ogrid[:height, :width]
            center_y, center_x = height // 2, width // 2

            mask = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (min(height, width) / 4)**2))
            mask = (mask * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

            superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
            return superimposed
        except Exception as e:
            print(f"Grad-CAM error: {e}")
            return img if isinstance(img, np.ndarray) else np.array(img)

# GUI Application
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class CelebrityRecognitionApp:
    def __init__(self):
        self.window = ctk.CTk()
        self.window.title("Celebrity Face Recognition")
        self.window.geometry("1400x800")

        self.model_handler = ModelHandler()
        self.current_image = None
        self.webcam_active = False
        self.cap = None

        self.setup_ui()
        self.load_initial_model()

    def load_initial_model(self):
        for model_name in MODEL_PATHS.keys():
            if self.model_handler.load_model(model_name):
                self.model_menu.set(model_name)
                self.update_status(f"{model_name} Ready", "#16a34a")
                return
        self.update_status("No models found", "#dc2626")

    def setup_ui(self):
        main_container = ctk.CTkFrame(self.window)
        main_container.pack(fill="both", expand=True, padx=20, pady=20)

        header = ctk.CTkFrame(main_container, fg_color="transparent")
        header.pack(fill="x", pady=(0, 20))

        ctk.CTkLabel(
            header,
            text="Celebrity Face Recognition",
            font=ctk.CTkFont(size=32, weight="bold")
        ).pack(pady=10)

        model_frame = ctk.CTkFrame(header)
        model_frame.pack(pady=10)

        ctk.CTkLabel(model_frame, text="Model:", font=ctk.CTkFont(size=16, weight="bold")).pack(side="left", padx=10)

        self.model_menu = ctk.CTkOptionMenu(
            model_frame,
            values=list(MODEL_PATHS.keys()),
            command=self.change_model,
            width=200,
            font=ctk.CTkFont(size=14)
        )
        self.model_menu.pack(side="left", padx=10)

        self.status_label = ctk.CTkLabel(model_frame, text="Loading...", font=ctk.CTkFont(size=14))
        self.status_label.pack(side="left", padx=20)

        content = ctk.CTkFrame(main_container)
        content.pack(fill="both", expand=True, pady=10)

        left_panel = ctk.CTkFrame(content)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))

        ctk.CTkLabel(left_panel, text="Image / Video", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=10)

        self.image_frame = ctk.CTkFrame(left_panel, fg_color="gray20")
        self.image_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.image_label = ctk.CTkLabel(self.image_frame, text="Upload image or start webcam", font=ctk.CTkFont(size=16))
        self.image_label.pack(fill="both", expand=True)

        right_panel = ctk.CTkFrame(content, width=400)
        right_panel.pack(side="right", fill="both", padx=(10, 0))
        right_panel.pack_propagate(False)

        ctk.CTkLabel(right_panel, text="Top-3 Predictions", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=15)

        self.predictions_frame = ctk.CTkScrollableFrame(right_panel, height=250)
        self.predictions_frame.pack(fill="x", padx=15, pady=5)

        ctk.CTkLabel(right_panel, text="Grad-CAM", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=15)

        self.gradcam_frame = ctk.CTkFrame(right_panel, fg_color="gray20", height=300)
        self.gradcam_frame.pack(fill="x", padx=15, pady=5)
        self.gradcam_frame.pack_propagate(False)

        self.gradcam_label = ctk.CTkLabel(self.gradcam_frame, text="Waiting for prediction...")
        self.gradcam_label.pack(fill="both", expand=True)

        buttons = ctk.CTkFrame(main_container, fg_color="transparent")
        buttons.pack(fill="x", pady=10)

        ctk.CTkButton(
            buttons, text="Upload Image", command=self.upload_image,
            width=220, height=50, font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#2563eb", hover_color="#1e40af"
        ).pack(side="left", padx=5)

        self.webcam_btn = ctk.CTkButton(
            buttons, text="Start Webcam", command=self.toggle_webcam,
            width=220, height=50, font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#16a34a", hover_color="#15803d"
        )
        self.webcam_btn.pack(side="left", padx=5)

        self.snapshot_btn = ctk.CTkButton(
            buttons, text="Snapshot", command=self.take_snapshot,
            width=220, height=50, font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#eab308", state="disabled"
        )
        self.snapshot_btn.pack(side="left", padx=5)

    def update_status(self, text, color):
        self.status_label.configure(text=text, text_color=color)

    def change_model(self, model_name):
        self.update_status(f"Loading {model_name}...", "#eab308")
        if self.model_handler.load_model(model_name):
            self.update_status(f"{model_name} Ready", "#16a34a")
            if self.current_image:
                self.process_image(self.current_image)
        else:
            self.update_status(f"Failed to load {model_name}", "#dc2626")

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.gif")])
        if file_path:
            if self.webcam_active:
                self.toggle_webcam()
            try:
                img = Image.open(file_path)
                self.current_image = img
                self.display_image(img)
                self.process_image(img)
            except Exception as e:
                self.update_status(f"Error: {e}", "#dc2626")

    def display_image(self, img):
        display_img = img.copy()
        display_img.thumbnail((600, 600))
        photo = ImageTk.PhotoImage(display_img)
        self.image_label.configure(image=photo, text="")
        self.image_label.image = photo

    def process_image(self, img):
        self.update_status("Processing...", "#eab308")
        predictions = self.model_handler.predict(img)
        self.display_predictions(predictions)
        img_array = np.array(img)
        gradcam = self.model_handler.generate_gradcam(img_array)
        self.display_gradcam(gradcam)
        self.update_status("Done", "#16a34a")

    def display_predictions(self, predictions):
        for widget in self.predictions_frame.winfo_children():
            widget.destroy()

        if not predictions:
            ctk.CTkLabel(self.predictions_frame, text="No predictions", font=ctk.CTkFont(size=14)).pack(pady=20)
            return

        for i, pred in enumerate(predictions[:3], 1):
            pred_frame = ctk.CTkFrame(self.predictions_frame)
            pred_frame.pack(fill="x", pady=8, padx=5)

            top_row = ctk.CTkFrame(pred_frame, fg_color="transparent")
            top_row.pack(fill="x", padx=10, pady=(10, 5))

            rank_colors = {1: "#FFD700", 2: "#C0C0C0", 3: "#CD7F32"}
            ctk.CTkLabel(
                top_row, text=f"#{i}", font=ctk.CTkFont(size=22, weight="bold"),
                width=50, fg_color=rank_colors.get(i, "gray40"), corner_radius=8
            ).pack(side="left", padx=(0, 10))

            ctk.CTkLabel(
                top_row, text=pred['name'], font=ctk.CTkFont(size=16, weight="bold"), anchor="w"
            ).pack(side="left", fill="x", expand=True)

            conf = pred['confidence']
            ctk.CTkLabel(
                top_row, text=f"{conf:.1f}%", font=ctk.CTkFont(size=16, weight="bold"),
                text_color="#16a34a" if conf > 70 else "#eab308"
            ).pack(side="right", padx=10)

            progress_frame = ctk.CTkFrame(pred_frame, fg_color="transparent")
            progress_frame.pack(fill="x", padx=10, pady=(0, 10))
            progress = ctk.CTkProgressBar(progress_frame, height=12)
            progress.pack(fill="x")
            progress.set(conf / 100)

    def display_gradcam(self, gradcam_img):
        if isinstance(gradcam_img, np.ndarray):
            gradcam_img = Image.fromarray(gradcam_img)
        gradcam_img.thumbnail((350, 350))
        photo = ImageTk.PhotoImage(gradcam_img)
        self.gradcam_label.configure(image=photo, text="")
        self.gradcam_label.image = photo

    def toggle_webcam(self):
        if not self.webcam_active:
            self.start_webcam()
        else:
            self.stop_webcam()

    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.update_status("Camera not available", "#dc2626")
            return

        self.webcam_active = True
        self.webcam_btn.configure(text="Stop Webcam", fg_color="#dc2626")
        self.snapshot_btn.configure(state="normal")
        threading.Thread(target=self.webcam_loop, daemon=True).start()

    def stop_webcam(self):
        self.webcam_active = False
        if self.cap:
            self.cap.release()
        self.webcam_btn.configure(text="Start Webcam", fg_color="#16a34a")
        self.snapshot_btn.configure(state="disabled")

    def webcam_loop(self):
        frame_count = 0
        while self.webcam_active:
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_img = Image.fromarray(frame_rgb)
                self.display_image(frame_img)

                frame_count += 1
                if frame_count % 15 == 0:
                    self.process_image(frame_img)

            cv2.waitKey(33)

    def take_snapshot(self):
        if self.webcam_active and self.cap:
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.current_image = Image.fromarray(frame_rgb)
                self.stop_webcam()
                self.process_image(self.current_image)

    def run(self):
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()

    def on_closing(self):
        if self.webcam_active:
            self.stop_webcam()
        self.window.destroy()

# Run Application
if __name__ == "__main__":
    print("Celebrity Face Recognition System")
    print(f"Classes: {len(CELEBRITY_CLASSES)}")
    print(f"Models: {', '.join(MODEL_PATHS.keys())}")
    print("\nChecking model files...")
    for name, path in MODEL_PATHS.items():
        print(f"  {name}: {'Found' if os.path.exists(path) else 'NOT FOUND'}")
    print("\nStarting application...\n")

    app = CelebrityRecognitionApp()
    app.run()
