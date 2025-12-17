import customtkinter as ctk
import cv2
from PIL import Image, ImageTk

class CelebrityGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Celebrity Face Classification")
        self.geometry("800x600")

   
        self.upload_button = ctk.CTkButton(self, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)

    
        self.cam_button = ctk.CTkButton(self, text="Start Webcam", command=self.start_webcam)
        self.cam_button.pack(pady=10)


        self.image_label = ctk.CTkLabel(self)
        self.image_label.pack()

        self.cap = None
        self.webcam_on = False

    def upload_image(self):
        from tkinter import filedialog
        file_path = filedialog.askopenfilename()
        if file_path:
            image = Image.open(file_path)
            image = image.resize((400, 400))
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo
           

    def start_webcam(self):
        if not self.webcam_on:
            self.cap = cv2.VideoCapture(0)
            self.webcam_on = True
            self.show_frame()
            self.cam_button.configure(text="Stop Webcam")
        else:
            self.webcam_on = False
            self.cap.release()
            self.cam_button.configure(text="Start Webcam")

    def show_frame(self):
        if self.webcam_on:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img = img.resize((400, 400))
                imgtk = ImageTk.PhotoImage(image=img)
                self.image_label.configure(image=imgtk)
                self.image_label.image = imgtk

            self.after(30, self.show_frame)  

if __name__ == "__main__":
    app = CelebrityGUI()
    app.mainloop()
