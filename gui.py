import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading

# إعدادات المظهر
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class CelebrityRecognitionApp:
    def __init__(self):
        self.window = ctk.CTk()
        self.window.title("Celebrity Face Recognition 🎬")
        self.window.geometry("1400x800")
        
        # متغيرات
        self.current_model = "ResNet50"
        self.webcam_active = False
        self.cap = None
        self.current_image = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # الـ Container الرئيسي
        main_container = ctk.CTkFrame(self.window)
        main_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # ═══════════════ الجزء العلوي - العنوان والتحكم ═══════════════
        header_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        header_frame.pack(fill="x", pady=(0, 20))
        
        title_label = ctk.CTkLabel(
            header_frame,
            text="🎬 Celebrity Face Recognition System",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        title_label.pack(pady=10)
        
        # ═══════════════ اختيار الموديل ═══════════════
        model_frame = ctk.CTkFrame(header_frame)
        model_frame.pack(pady=10)
        
        ctk.CTkLabel(
            model_frame,
            text="Choose Model:",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(side="left", padx=10)
        
        self.model_menu = ctk.CTkOptionMenu(
            model_frame,
            values=["ResNet50", "VGG16", "Compare Both"],
            command=self.change_model,
            width=200,
            font=ctk.CTkFont(size=14)
        )
        self.model_menu.pack(side="left", padx=10)
        
        # ═══════════════ الجزء الأوسط - عرض الصورة والنتائج ═══════════════
        content_frame = ctk.CTkFrame(main_container)
        content_frame.pack(fill="both", expand=True, pady=10)
        
        # الجانب الأيسر - الصورة
        left_panel = ctk.CTkFrame(content_frame)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        ctk.CTkLabel(
            left_panel,
            text="📷 Image / Video Feed",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=10)
        
        self.image_label = ctk.CTkLabel(left_panel, text="")
        self.image_label.pack(fill="both", expand=True, padx=10, pady=10)
        
        # الجانب الأيمن - النتائج
        right_panel = ctk.CTkFrame(content_frame, width=400)
        right_panel.pack(side="right", fill="both", padx=(10, 0))
        right_panel.pack_propagate(False)
        
        # Top-3 Predictions
        ctk.CTkLabel(
            right_panel,
            text="🏆 Top-3 Predictions",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=15)
        
        self.predictions_frame = ctk.CTkScrollableFrame(right_panel, height=200)
        self.predictions_frame.pack(fill="x", padx=15, pady=5)
        
        # Grad-CAM
        ctk.CTkLabel(
            right_panel,
            text="🔥 Grad-CAM Visualization",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=15)
        
        self.gradcam_label = ctk.CTkLabel(right_panel, text="")
        self.gradcam_label.pack(padx=15, pady=5)
        
        # Model Comparison (إذا اختار Compare Both)
        self.comparison_frame = ctk.CTkFrame(right_panel)
        
        # ═══════════════ الأزرار السفلية ═══════════════
        buttons_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        buttons_frame.pack(fill="x", pady=10)
        
        # زر رفع صورة
        upload_btn = ctk.CTkButton(
            buttons_frame,
            text="📁 Upload Image",
            command=self.upload_image,
            width=200,
            height=50,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#2563eb",
            hover_color="#1e40af"
        )
        upload_btn.pack(side="left", padx=10)
        
        # زر الكاميرا
        self.webcam_btn = ctk.CTkButton(
            buttons_frame,
            text="📹 Start Webcam",
            command=self.toggle_webcam,
            width=200,
            height=50,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#16a34a",
            hover_color="#15803d"
        )
        self.webcam_btn.pack(side="left", padx=10)
        
        # زر Confusion Matrix
        confusion_btn = ctk.CTkButton(
            buttons_frame,
            text="📊 Show Confusion Matrix",
            command=self.show_confusion_matrix,
            width=200,
            height=50,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#7c3aed",
            hover_color="#6d28d9"
        )
        confusion_btn.pack(side="left", padx=10)
        
        # زر Accuracy
        accuracy_btn = ctk.CTkButton(
            buttons_frame,
            text="📈 Show Accuracy",
            command=self.show_accuracy,
            width=200,
            height=50,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#dc2626",
            hover_color="#b91c1c"
        )
        accuracy_btn.pack(side="left", padx=10)
        
    def change_model(self, choice):
        self.current_model = choice
        print(f"Model changed to: {choice}")
        
        # إذا اختار Compare Both، أظهر إطار المقارنة
        if choice == "Compare Both":
            self.comparison_frame.pack(fill="x", padx=15, pady=15)
            self.update_comparison_display()
        else:
            self.comparison_frame.pack_forget()
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            # إيقاف الكاميرا إذا كانت شغالة
            if self.webcam_active:
                self.toggle_webcam()
            
            # تحميل وعرض الصورة
            image = Image.open(file_path)
            self.current_image = image
            self.display_image(image)
            
            # عمل prediction (هنا تستدعي موديلك)
            self.predict_image(image)
    
    def display_image(self, image):
        # تغيير حجم الصورة للعرض
        display_image = image.copy()
        display_image.thumbnail((600, 600))
        
        photo = ImageTk.PhotoImage(display_image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo
    
    def predict_image(self, image):
        # هنا تحط كود الـ prediction بتاعك
        # مثال:
        predictions = self.get_predictions(image)
        self.display_predictions(predictions)
        
        # Grad-CAM
        gradcam_image = self.generate_gradcam(image)
        self.display_gradcam(gradcam_image)
    
    def get_predictions(self, image):
        # هنا تحط كود الموديل الحقيقي
        # مثال توضيحي فقط:
        sample_predictions = [
            {"name": "Brad Pitt", "confidence": 95.8},
            {"name": "Leonardo DiCaprio", "confidence": 78.3},
            {"name": "Tom Cruise", "confidence": 65.1}
        ]
        return sample_predictions
    
    def display_predictions(self, predictions):
        # مسح النتائج القديمة
        for widget in self.predictions_frame.winfo_children():
            widget.destroy()
        
        # عرض Top-3
        for i, pred in enumerate(predictions[:3], 1):
            pred_frame = ctk.CTkFrame(self.predictions_frame)
            pred_frame.pack(fill="x", pady=5)
            
            # الترتيب
            rank_label = ctk.CTkLabel(
                pred_frame,
                text=f"#{i}",
                font=ctk.CTkFont(size=20, weight="bold"),
                width=40
            )
            rank_label.pack(side="left", padx=5)
            
            # الاسم
            name_label = ctk.CTkLabel(
                pred_frame,
                text=pred["name"],
                font=ctk.CTkFont(size=16),
                anchor="w"
            )
            name_label.pack(side="left", fill="x", expand=True, padx=5)
            
            # Confidence
            conf_label = ctk.CTkLabel(
                pred_frame,
                text=f"{pred['confidence']:.1f}%",
                font=ctk.CTkFont(size=16, weight="bold"),
                text_color="#22c55e" if pred['confidence'] > 80 else "#eab308"
            )
            conf_label.pack(side="right", padx=5)
            
            # Progress bar
            progress = ctk.CTkProgressBar(pred_frame, width=200)
            progress.pack(side="right", padx=5)
            progress.set(pred["confidence"] / 100)
    
    def generate_gradcam(self, image):
        # هنا تحط كود Grad-CAM الحقيقي
        # مثال توضيحي:
        return image  # إرجاع الصورة الأصلية كمثال
    
    def display_gradcam(self, gradcam_image):
        display_image = gradcam_image.copy()
        display_image.thumbnail((350, 350))
        
        photo = ImageTk.PhotoImage(display_image)
        self.gradcam_label.configure(image=photo)
        self.gradcam_label.image = photo
    
    def update_comparison_display(self):
        # مسح المحتوى القديم
        for widget in self.comparison_frame.winfo_children():
            widget.destroy()
        
        ctk.CTkLabel(
            self.comparison_frame,
            text="⚖️ Model Comparison",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        # مثال مقارنة
        models_data = [
            {"name": "ResNet50", "accuracy": 94.5, "time": "0.12s"},
            {"name": "VGG16", "accuracy": 92.8, "time": "0.18s"}
        ]
        
        for model in models_data:
            model_frame = ctk.CTkFrame(self.comparison_frame)
            model_frame.pack(fill="x", padx=10, pady=5)
            
            ctk.CTkLabel(
                model_frame,
                text=model["name"],
                font=ctk.CTkFont(size=14, weight="bold")
            ).pack(side="left", padx=10)
            
            ctk.CTkLabel(
                model_frame,
                text=f"Acc: {model['accuracy']}%"
            ).pack(side="left", padx=10)
            
            ctk.CTkLabel(
                model_frame,
                text=f"Time: {model['time']}"
            ).pack(side="right", padx=10)
    
    def toggle_webcam(self):
        if not self.webcam_active:
            self.start_webcam()
        else:
            self.stop_webcam()
    
    def start_webcam(self):
        self.webcam_active = True
        self.webcam_btn.configure(text="⏹️ Stop Webcam", fg_color="#dc2626")
        
        self.cap = cv2.VideoCapture(0)
        threading.Thread(target=self.webcam_loop, daemon=True).start()
    
    def stop_webcam(self):
        self.webcam_active = False
        self.webcam_btn.configure(text="📹 Start Webcam", fg_color="#16a34a")
        
        if self.cap:
            self.cap.release()
    
    def webcam_loop(self):
        while self.webcam_active:
            ret, frame = self.cap.read()
            if ret:
                # تحويل BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # عمل prediction على الـ frame
                # هنا تحط كود الـ real-time detection
                predictions = self.get_predictions_from_frame(frame_rgb)
                
                # رسم النتائج على الفريم
                frame_with_text = self.draw_predictions_on_frame(frame_rgb, predictions)
                
                # عرض الفريم
                image = Image.fromarray(frame_with_text)
                self.display_image(image)
                
                # تحديث النتائج
                self.display_predictions(predictions)
                
                # تأخير صغير
                self.window.after(10)
    
    def get_predictions_from_frame(self, frame):
        # هنا تحط كود الـ prediction من الفريم
        # مثال:
        return [
            {"name": "Unknown", "confidence": 0.0}
        ]
    
    def draw_predictions_on_frame(self, frame, predictions):
        # رسم النتائج على الفريم
        frame_copy = frame.copy()
        
        if predictions and predictions[0]["confidence"] > 50:
            # رسم مستطيل ونص (مثال)
            cv2.putText(
                frame_copy,
                f"{predictions[0]['name']}: {predictions[0]['confidence']:.1f}%",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
        
        return frame_copy
    
    def show_confusion_matrix(self):
        # نافذة جديدة لعرض Confusion Matrix
        matrix_window = ctk.CTkToplevel(self.window)
        matrix_window.title("Confusion Matrix")
        matrix_window.geometry("700x700")
        
        ctk.CTkLabel(
            matrix_window,
            text="📊 Confusion Matrix",
            font=ctk.CTkFont(size=24, weight="bold")
        ).pack(pady=20)
        
        # هنا تحط الـ Confusion Matrix الحقيقية
        info_label = ctk.CTkLabel(
            matrix_window,
            text="Model: " + self.current_model + "\n\n" +
                 "Here you can display the confusion matrix\n" +
                 "using matplotlib or seaborn",
            font=ctk.CTkFont(size=14)
        )
        info_label.pack(pady=20)
    
    def show_accuracy(self):
        # نافذة جديدة لعرض Accuracy
        accuracy_window = ctk.CTkToplevel(self.window)
        accuracy_window.title("Model Accuracy")
        accuracy_window.geometry("600x500")
        
        ctk.CTkLabel(
            accuracy_window,
            text="📈 Model Performance",
            font=ctk.CTkFont(size=24, weight="bold")
        ).pack(pady=20)
        
        # عرض معلومات الدقة
        info_frame = ctk.CTkFrame(accuracy_window)
        info_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        metrics = [
            {"name": "Accuracy", "value": "94.5%"},
            {"name": "Precision", "value": "93.2%"},
            {"name": "Recall", "value": "95.1%"},
            {"name": "F1-Score", "value": "94.1%"}
        ]
        
        for metric in metrics:
            metric_frame = ctk.CTkFrame(info_frame)
            metric_frame.pack(fill="x", pady=10, padx=10)
            
            ctk.CTkLabel(
                metric_frame,
                text=metric["name"],
                font=ctk.CTkFont(size=18, weight="bold")
            ).pack(side="left", padx=20)
            
            ctk.CTkLabel(
                metric_frame,
                text=metric["value"],
                font=ctk.CTkFont(size=18),
                text_color="#22c55e"
            ).pack(side="right", padx=20)
    
    def run(self):
        self.window.mainloop()

# تشغيل التطبيق
if __name__ == "__main__":
    app = CelebrityRecognitionApp()
    app.run()
