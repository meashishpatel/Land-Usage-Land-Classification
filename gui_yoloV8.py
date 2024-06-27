import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar
from PIL import Image, ImageTk, ImageDraw, ImageFont
from ultralytics import YOLO
import os
import numpy as np
from osgeo import gdal

class YOLOGUI:
    def __init__(self, root):
        self.root = root

        # Initialize YOLO model
        self.model = YOLO('runs/segment/train21/weights/best.pt')

        # Read and initialize results counter
        self.counter_file = "counter.txt"
        self.results_counter = self.read_results_counter()

        # Buttons
        self.select_button = tk.Button(self.root, text="Select Image", command=self.select_image)
        self.select_button.pack(pady=10)

        self.proceed_button = tk.Button(self.root, text="Proceed", command=self.proceed_with_prediction)
        self.proceed_button.pack(pady=10)

        self.save_button = tk.Button(self.root, text="Save Image", command=self.save_image, state=tk.DISABLED)
        self.save_button.pack(pady=10)

        # Image display
        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=10)

        # Progress bar with text
        self.progress_label = tk.Label(self.root, text="")
        self.progress_label.pack(pady=10)

        self.progress_bar = Progressbar(self.root, length=200, mode='determinate')
        self.progress_bar.pack(pady=10)

        self.prediction_result_label = tk.Label(self.root, text="Prediction Result:")
        self.prediction_result_label.pack()

        self.result_text = tk.Text(self.root, height=5, width=50)
        self.result_text.pack()

        # Initialize variables
        self.image_path = ""
        self.predicted_image_path = ""
        self.prediction_speed = ""
        self.prediction_classes = {}

    def read_results_counter(self):
        try:
            with open(self.counter_file, 'r') as f:
                counter = int(f.read().strip())
        except FileNotFoundError:
            counter = 1  # Initialize counter if file does not exist
        return counter

    def write_results_counter(self, counter):
        with open(self.counter_file, 'w') as f:
            f.write(str(counter))

    def select_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.tif;*.tiff")])
        if self.image_path:
            if self.image_path.lower().endswith(('.tif', '.tiff')):
                self.display_message("Selected TIFF image, not displaying due to large size.")
            else:
                self.display_image(self.image_path)
            self.save_button.config(state=tk.DISABLED)  # Disable save button until prediction is done

    def proceed_with_prediction(self):
        if not self.image_path:
            messagebox.showerror("Error", "Please select an image first.")
            return

        self.progress_label.config(text="Predicting...")
        self.progress_bar['value'] = 50
        self.progress_bar.update()

        # Handle TIFF images separately
        if self.image_path.lower().endswith(('.tif', '.tiff')):
            dataset = gdal.Open(self.image_path)
            band = dataset.GetRasterBand(1)
            image = band.ReadAsArray()

            # Ensure the image is in a format YOLO can process
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[2] > 3:
                image = image[:, :, :3]
            results = self.model.predict(image, save_conf=True, show_boxes=False, save=True)
        else:
            results = self.model.predict(self.image_path, save_conf=True, show_boxes=False, save=True)

        self.progress_bar['value'] = 100
        self.progress_label.config(text="Prediction Complete")
        self.progress_bar.update()

        # Update prediction result
        self.result_text.delete(1.0, tk.END)
        self.prediction_speed = []
        self.prediction_classes = {}

        if isinstance(results, list):
            total_images = len(results)
            total_speed = 0

            for result in results:
                # Update speed for each image
                speed_info = result.speed
                self.prediction_speed.append(speed_info)

                # Calculate and aggregate total speed
                total_speed += speed_info['inference']

                # Update detection counts for each class
                for pred in result.boxes:
                    cls = int(pred.cls.item())
                    if cls in self.prediction_classes:
                        self.prediction_classes[cls] += 1
                    else:
                        self.prediction_classes[cls] = 1

            # Calculate average speed across all images
            avg_speed = total_speed / total_images

        else:
            # Handle single result case
            speed_info = results.speed
            self.prediction_speed = speed_info

            # Calculate total speed
            total_speed = speed_info['inference']

            # Update detection counts for single result
            for pred in results.boxes:
                cls = int(pred.cls.item())
                if cls in self.prediction_classes:
                    self.prediction_classes[cls] += 1
                else:
                    self.prediction_classes[cls] = 1

            # Set average speed for single image case
            avg_speed = total_speed

        # Update GUI with prediction results
        self.result_text.delete(1.0, tk.END)
        for cls, count in self.prediction_classes.items():
            self.result_text.insert(tk.END, f"Class: {cls}, Count: {count}\n")
        
        self.result_text.insert(tk.END, f"Avg Inference Time: {avg_speed:.2f} ms\n")

        # Enable save button
        self.save_button.config(state=tk.NORMAL)

        # Get the path of the saved image
        self.predicted_image_path = results[0].path if results else ""

        # Increment and store results counter
        self.results_counter += 1
        self.write_results_counter(self.results_counter)


    def display_image(self, image_path):
        image = Image.open(image_path)
        image = image.resize((400, 300), Image.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo

    def display_message(self, message):
        self.image_label.configure(text=message)

    def save_image(self):
        if not self.predicted_image_path:
            messagebox.showerror("Error", "No predicted image to save.")
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if save_path:
            # Add legend and details to the image
            self.add_legend_and_details_to_image(self.predicted_image_path, save_path)
            messagebox.showinfo("Image Saved", f"Predicted image saved successfully at {save_path}.")

    def add_legend_and_details_to_image(self, image_path, save_path):
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        legend = [
            "0: car",
            "1: forest",
            "2: road",
            "3: building",
            "4: water",
            "5: grass",
            "6: person"
        ]

        # Position for the legend
        x, y = 10, 10
        for item in legend:
            draw.text((x, y), item, font=font, fill="white")
            y += 15

        # Add prediction details
        y += 15
        draw.text((x, y), f"Speed: {self.prediction_speed}", font=font, fill="white")
        y += 15
        for cls, count in self.prediction_classes.items():
            draw.text((x, y), f"Class {cls}: {count}", font=font, fill="white")
            y += 15

        image.save(save_path)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("YOLO Image Prediction")
    gui = YOLOGUI(root)
    root.mainloop()
