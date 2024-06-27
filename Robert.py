import numpy as np
import cv2
from tkinter import Tk, Label, Button, filedialog, messagebox, Canvas, Frame, Scrollbar
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from osgeo import gdal
import time

class RobertEdgeDetectionApp:
    def __init__(self, parent):
        self.parent = parent
        self.frame = Frame(self.parent, bg="white")
        self.frame.pack(fill='both', expand=True)

        self.input_image_path = None
        self.input_image = None
        self.filtered_image = None
        self.edge_detected_image = None

        self.RESOLUTION_THRESHOLD = 1920 * 1080  # Example resolution threshold for display purposes

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

        self.btn_proceed = Button(self.scrollable_frame, text="Detect Edges", command=self.edge_detection)
        self.btn_proceed.pack(pady=20)

        self.btn_save_image = Button(self.scrollable_frame, text="Save Image", command=self.save_image)
        self.btn_save_image.pack(pady=10)

        # Pack the canvas and scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

    def resize_image(self, image, max_size=(600, 600)):
        h, w = image.shape[:2]
        if h > max_size[1] or w > max_size[0]:
            scaling_factor = min(max_size[0] / w, max_size[1] / h)
            new_size = (int(w * scaling_factor), int(h * scaling_factor))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        return image

    def ensure_three_channels(self, image):
        if image.shape[2] > 3:
            return image[:, :, :3]
        elif image.shape[2] == 1:
            return np.repeat(image, 3, axis=2)
        else:
            return image

    def select_image(self):
        self.input_image_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.tif;*.bmp;*.tiff")])
        if self.input_image_path:
            self.lbl_image_path.config(text=self.input_image_path)
            try:
                print(f"Loading image: {self.input_image_path}")
                start_time = time.time()
                dataset = gdal.Open(self.input_image_path)
                self.input_image = dataset.ReadAsArray()
                print(f"Image shape (initial): {self.input_image.shape}")

                if len(self.input_image.shape) == 3:
                    self.input_image = np.transpose(self.input_image, (1, 2, 0))  # GDAL returns images as (bands, rows, cols)
                else:
                    self.input_image = np.stack([self.input_image] * 3, axis=-1)  # Handle grayscale images

                self.input_image = self.ensure_three_channels(self.input_image)
                print(f"Image shape (processed): {self.input_image.shape}")

                if self.input_image.shape[0] * self.input_image.shape[1] > self.RESOLUTION_THRESHOLD:
                    messagebox.showinfo("Large Image", "The selected image is too large to display. It will be processed directly.")
                else:
                    self.display_image(self.input_image, self.lbl_input_image)
                end_time = time.time()
                print(f"Image loaded and displayed in {end_time - start_time} seconds")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def display_image(self, image, label):
        resized_image = self.resize_image(image)
        image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_tk = ImageTk.PhotoImage(image_pil)
        label.config(image=image_tk)
        label.image = image_tk

    def edge_detection(self):
        if self.input_image is None:
            messagebox.showerror("Error", "Please select an image first.")
            return

        try:
            print("Starting edge detection...")
            start_time = time.time()

            gray_image = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2GRAY)
            gray_image = gray_image.astype(np.float64)
            self.filtered_image = np.zeros_like(gray_image)

            # Robert Operator Masks
            Mx = np.array([[1, 0], [0, -1]])
            My = np.array([[0, 1], [-1, 0]])

            # Process the image in chunks to avoid memory issues
            chunk_size = 1000  # Define chunk size
            for i in range(0, gray_image.shape[0] - 1, chunk_size):
                for j in range(0, gray_image.shape[1] - 1, chunk_size):
                    i_end = min(i + chunk_size, gray_image.shape[0] - 1)
                    j_end = min(j + chunk_size, gray_image.shape[1] - 1)

                    # Edge Detection Process for the chunk
                    for x in range(i, i_end):
                        for y in range(j, j_end):
                            Gx = np.sum(Mx * gray_image[x:x+2, y:y+2])
                            Gy = np.sum(My * gray_image[x:x+2, y:y+2])
                            self.filtered_image[x, y] = np.sqrt(Gx**2 + Gy**2)

            self.filtered_image = np.uint8(self.filtered_image)

            # Define a threshold value
            threshold_value = 100
            self.edge_detected_image = np.where(self.filtered_image > threshold_value, 255, 0).astype(np.uint8)

            self.display_image(cv2.cvtColor(self.edge_detected_image, cv2.COLOR_GRAY2BGR), self.lbl_output_image)

            # Display images using matplotlib
            plt.figure(figsize=(15, 5))

            plt.subplot(131), plt.imshow(cv2.cvtColor(self.input_image, cv2.COLOR_BGR2RGB))
            plt.title('Input Image'), plt.xticks([]), plt.yticks([])

            plt.subplot(132), plt.imshow(self.filtered_image, cmap='gray')
            plt.title('Filtered Image'), plt.xticks([]), plt.yticks([])

            plt.subplot(133), plt.imshow(self.edge_detected_image, cmap='gray')
            plt.title('Edge Detected Image'), plt.xticks([]), plt.yticks([])

            plt.tight_layout()
            plt.show()

            end_time = time.time()
            print(f"Edge detection completed in {end_time - start_time} seconds")
        except Exception as e:
            messagebox.showerror("Error", f"Edge detection failed: {str(e)}")

    def save_image(self):
        if self.edge_detected_image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
            if file_path:
                cv2.imwrite(file_path, self.edge_detected_image)
                messagebox.showinfo("Success", f"Image saved as {file_path}")
        else:
            messagebox.showerror("Error", "No image to save.")