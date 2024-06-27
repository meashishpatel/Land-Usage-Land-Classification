import cv2
import numpy as np
from tkinter import Tk, Label, Button, Scale, HORIZONTAL, filedialog, messagebox, Canvas, Frame, Scrollbar
from tkinter.ttk import Progressbar
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import threading

class ThresholdingApp:
    def __init__(self, parent):
        self.parent = parent
        self.frame = Frame(self.parent, bg="white")
        self.frame.pack(fill='both', expand=True)

        self.input_image_path = None
        self.input_image = None
        self.thresholded_image = None

        self.create_widgets()

    def create_widgets(self):
        self.canvas = Canvas(self.frame)
        self.scrollbar = Scrollbar(self.frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Add widgets to the scrollable frame
        self.btn_upload_image = Button(self.scrollable_frame, text="Upload Image", command=self.select_image)
        self.btn_upload_image.pack(pady=10)

        self.lbl_image_path = Label(self.scrollable_frame, text="No file selected")
        self.lbl_image_path.pack(pady=5)

        self.lbl_input_image = Label(self.scrollable_frame)
        self.lbl_input_image.pack(pady=10)

        self.lbl_output_image = Label(self.scrollable_frame)
        self.lbl_output_image.pack(pady=10)

        self.threshold_slider = Scale(self.scrollable_frame, from_=0, to=255, orient=HORIZONTAL, label="Threshold Value")
        self.threshold_slider.pack(pady=20)

        self.btn_process_image = Button(self.scrollable_frame, text="Process Image", command=self.start_thresholding)
        self.btn_process_image.pack(pady=20)

        self.progress = Progressbar(self.scrollable_frame, orient=HORIZONTAL, length=400, mode='determinate')
        self.progress.pack(pady=20)

        self.btn_save_image = Button(self.scrollable_frame, text="Save Image", command=self.save_image)
        self.btn_save_image.pack(pady=10)

        # Pack the canvas and scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

    def select_image(self):
        self.input_image_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.tif;*.bmp;*.tiff")])
        if self.input_image_path:
            self.lbl_image_path.config(text=self.input_image_path)
            self.input_image = cv2.imread(self.input_image_path, cv2.IMREAD_GRAYSCALE)
            if self.input_image is None:
                messagebox.showerror("Error", "Failed to load image.")
                return

            self.display_image(self.input_image, self.lbl_input_image)

    def display_image(self, image, label):
        max_size = (600, 600)
        image_pil = Image.fromarray(image)
        image_pil.thumbnail(max_size, Image.LANCZOS)  # Resize while maintaining aspect ratio
        image_tk = ImageTk.PhotoImage(image_pil)
        label.config(image=image_tk)
        label.image = image_tk

    def start_thresholding(self):
        threading.Thread(target=self.apply_threshold).start()

    def apply_threshold(self):
        if self.input_image is None:
            messagebox.showerror("Error", "Please select an image first.")
            return

        threshold_value = self.threshold_slider.get()
        height, width = self.input_image.shape

        self.thresholded_image = np.zeros_like(self.input_image)
        
        self.progress["value"] = 0
        self.progress["maximum"] = height

        for i in range(height):
            for j in range(width):
                if self.input_image[i, j] > threshold_value:
                    self.thresholded_image[i, j] = 255
                else:
                    self.thresholded_image[i, j] = 0

            self.progress["value"] = i + 1
            self.parent.update_idletasks()

        self.display_image(self.thresholded_image, self.lbl_output_image)

        # Display images using matplotlib
        plt.figure(figsize=(10, 5))

        plt.subplot(121), plt.imshow(self.input_image, cmap='gray')
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])

        plt.subplot(122), plt.imshow(self.thresholded_image, cmap='gray')
        plt.title('Thresholded Image'), plt.xticks([]), plt.yticks([])

        plt.tight_layout()
        plt.show()

    def save_image(self):
        if self.thresholded_image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
            if file_path:
                cv2.imwrite(file_path, self.thresholded_image)
                messagebox.showinfo("Success", f"Image saved as {file_path}")
        else:
            messagebox.showerror("Error", "No image to save.")