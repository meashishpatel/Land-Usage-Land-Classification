import numpy as np
import cv2
from tkinter import Tk, Label, Button, filedialog, messagebox, Frame
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import tifffile as tiff

class OtsusSegmentationApp:
    def __init__(self, parent):
        self.parent = parent
        self.frame = Frame(self.parent, bg="white")
        self.frame.pack(fill='both', expand=True)
        
        self.input_image_path = None
        self.input_image = None
        self.segmented_image = None
        self.thresh_image = None

        self.RESOLUTION_THRESHOLD = 5000 * 5000  # Increased threshold for larger images

        self.create_widgets()

    def create_widgets(self):
        self.btn_upload_image = Button(self.frame, text="Upload Image", command=self.select_image)
        self.btn_upload_image.pack(pady=10)

        self.lbl_image_path = Label(self.frame, text="No file selected")
        self.lbl_image_path.pack(pady=5)

        self.lbl_input_image = Label(self.frame)
        self.lbl_input_image.pack(pady=10)

        self.lbl_output_image = Label(self.frame)
        self.lbl_output_image.pack(pady=10)

        self.btn_proceed = Button(self.frame, text="Segment Image", command=self.segment_image)
        self.btn_proceed.pack(pady=20)

        self.btn_save_image = Button(self.frame, text="Save Image", command=self.segment_and_save_image)
        self.btn_save_image.pack(pady=10)

    def resize_image(self, image, max_size=(800, 600)):
        h, w = image.shape[:2]
        if h > max_size[1] or w > max_size[0]:
            scaling_factor = min(max_size[0] / w, max_size[1] / h)
            new_size = (int(w * scaling_factor), int(h * scaling_factor))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        return image

    def select_image(self):
        self.input_image_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.tif;*.bmp")])
        if self.input_image_path:
            self.lbl_image_path.config(text=self.input_image_path)
            try:
                if self.input_image_path.lower().endswith(('.tif', '.tiff')):
                    tif_image = tiff.imread(self.input_image_path)
                    if tif_image.shape[-1] > 3:
                        tif_image = tif_image[..., :3]
                    self.input_image = tif_image
                else:
                    pil_image = Image.open(self.input_image_path)
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    self.input_image = np.array(pil_image)
                    self.input_image = cv2.cvtColor(self.input_image, cv2.COLOR_RGB2BGR)

                self.display_image(self.input_image, self.lbl_input_image)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def display_image(self, image, label):
        resized_image = self.resize_image(image)
        image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_tk = ImageTk.PhotoImage(image_pil)
        label.config(image=image_tk)
        label.image = image_tk

    def segment_and_save_image(self):
        if self.input_image is None:
            messagebox.showerror("Error", "Please select an image first.")
            return

        gray = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        self.segmented_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        self.thresh_image = thresh

        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if file_path:
            cv2.imwrite(file_path, self.segmented_image)
            messagebox.showinfo("Success", f"Image saved as {file_path}")

    def segment_image(self):
        if self.input_image is None:
            messagebox.showerror("Error", "Please select an image first.")
            return

        gray = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        self.segmented_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        self.thresh_image = thresh

        self.display_image(self.segmented_image, self.lbl_output_image)

        cv2.imwrite('thresh.png', thresh)

        plt.figure(figsize=(10, 5))
        b, g, r = cv2.split(self.input_image)
        rgb_img = cv2.merge([r, g, b])

        plt.subplot(121), plt.imshow(rgb_img)
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])

        plt.subplot(122), plt.imshow(thresh, 'gray')
        plt.title("Otsu's binary threshold"), plt.xticks([]), plt.yticks([])

        plt.tight_layout()
        plt.show()