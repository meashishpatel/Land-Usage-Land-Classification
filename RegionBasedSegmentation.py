import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Label, Button, filedialog, messagebox, Canvas, Frame, Scrollbar
from PIL import Image, ImageTk
import cv2
from skimage.filters import sobel
from skimage.segmentation import watershed
from skimage.color import label2rgb
from scipy import ndimage as ndi

class WatershedSegmentationApp:
    def __init__(self, parent):
        self.parent = parent
        self.frame = Frame(self.parent, bg="white")
        self.frame.pack(fill='both', expand=True)

        self.input_image_path = None
        self.input_image = None
        self.elevation_map = None
        self.markers = None
        self.segmentation = None
        self.image_label_overlay = None

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

        self.btn_perform_sobel = Button(self.scrollable_frame, text="Perform Sobel Edge Detection", command=self.perform_sobel)
        self.btn_perform_sobel.pack(pady=10)

        self.btn_create_markers = Button(self.scrollable_frame, text="Create Markers", command=self.create_markers)
        self.btn_create_markers.pack(pady=10)

        self.btn_perform_watershed = Button(self.scrollable_frame, text="Perform Watershed Segmentation", command=self.perform_watershed)
        self.btn_perform_watershed.pack(pady=10)

        self.lbl_output_image = Label(self.scrollable_frame)
        self.lbl_output_image.pack(pady=10)

        # Pack the canvas and scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

    def select_image(self):
        self.input_image_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.tif;*.bmp")])
        if self.input_image_path:
            self.lbl_image_path.config(text=self.input_image_path)
            self.input_image = cv2.imread(self.input_image_path)
            self.display_image(self.input_image, self.lbl_input_image)

    def display_image(self, image, label):
        max_size = (600, 600)  # Maximum size for the display area
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_pil.thumbnail(max_size, Image.LANCZOS)  # Resize while maintaining aspect ratio
        image_tk = ImageTk.PhotoImage(image_pil)
        label.config(image=image_tk)
        label.image = image_tk

    def perform_sobel(self):
        if self.input_image is None:
            messagebox.showerror("Error", "Please select an image first.")
            return

        try:
            # Convert to grayscale
            gray_image = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2GRAY)

            # Perform Sobel edge detection
            self.elevation_map = sobel(gray_image)

            # Display edges image
            self.display_image(self.elevation_map.astype(np.uint8) * 255, self.lbl_output_image)
        except Exception as e:
            messagebox.showerror("Error", f"Error during edge detection: {str(e)}")

    def create_markers(self):
        if self.elevation_map is None:
            messagebox.showerror("Error", "Please perform Sobel edge detection first.")
            return

        try:
            # Create markers based on elevation map
            self.markers = np.zeros_like(self.elevation_map)
            self.markers[self.elevation_map < 0.1 * self.elevation_map.max()] = 1
            self.markers[self.elevation_map > 0.3 * self.elevation_map.max()] = 2

            # Display markers image
            self.display_image(self.markers.astype(np.uint8) * 255, self.lbl_output_image)
        except Exception as e:
            messagebox.showerror("Error", f"Error creating markers: {str(e)}")

    def perform_watershed(self):
        if self.markers is None:
            messagebox.showerror("Error", "Please create markers first.")
            return

        try:
            # Perform watershed segmentation
            self.segmentation = watershed(self.elevation_map, self.markers)

            # Fill holes in segmentation
            self.segmentation = ndi.binary_fill_holes(self.segmentation - 1)

            # Label regions and create overlay
            labeled_coins, _ = ndi.label(self.segmentation)
            self.image_label_overlay = label2rgb(labeled_coins, image=self.input_image)

            # Display segmented image
            self.display_image((self.image_label_overlay * 255).astype(np.uint8), self.lbl_output_image)
        except Exception as e:
            messagebox.showerror("Error", f"Error during watershed segmentation: {str(e)}")