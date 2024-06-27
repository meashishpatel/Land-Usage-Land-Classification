import tkinter as tk
from tkinter import filedialog, messagebox
from osgeo import gdal
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import ConvergenceWarning
from tkinter.ttk import Progressbar
import threading

class RandomForestApp:
    def __init__(self, parent):
        self.parent = parent
        self.frame = tk.Frame(self.parent, bg="white")
        self.frame.pack(fill=tk.BOTH, expand=True)
        self.gcp_path = None  # Initialize gcp_path here
        self.create_widgets()

    def create_widgets(self):
        global progress_text

        # UAV image upload button
        self.btn_upload_image = tk.Button(self.frame, text="Upload UAV Image", command=self.select_image)
        self.btn_upload_image.pack(pady=10)

        self.lbl_image_path = tk.Label(self.frame, text="No file selected")
        self.lbl_image_path.pack(pady=5)

        # GCP file upload option
        self.gcp_option = tk.StringVar(value="No")
        tk.Label(self.frame, text="Do you want to upload Ground Control Points (GCP)?").pack(pady=5)
        tk.Radiobutton(self.frame, text="Yes", variable=self.gcp_option, value="Yes", command=self.toggle_gcp_option).pack()
        tk.Radiobutton(self.frame, text="No", variable=self.gcp_option, value="No", command=self.toggle_gcp_option).pack()

        # GCP file upload button (initially hidden)
        self.btn_upload_gcp = tk.Button(self.frame, text="Upload GCP CSV File", command=self.select_gcp)
        self.lbl_gcp_path = tk.Label(self.frame, text="No file selected")

        # Progress text
        progress_text = tk.StringVar()
        self.lbl_progress = tk.Label(self.frame, textvariable=progress_text)
        self.lbl_progress.pack(pady=5)

        # Progress bar
        self.progress_bar = Progressbar(self.frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress_bar.pack(pady=20)

        # Proceed button
        self.btn_proceed = tk.Button(self.frame, text="Proceed", command=self.start_processing)
        self.btn_proceed.pack(pady=20)

        # Save buttons (initially hidden)
        self.btn_save_png = tk.Button(self.frame, text="Save Classified Image as PNG", command=self.save_classified_image_png)
        self.btn_save_tif = tk.Button(self.frame, text="Save Classified Image as TIF", command=self.save_classified_image_tif)

    def display_image(self, image_data, title='Image'):
        plt.figure(figsize=(10, 10))
        plt.imshow(image_data)
        plt.title(title)
        plt.show()

    def save_classified_image_png(self):
        global color_image
        if color_image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
            if file_path:
                plt.figure(figsize=(10, 10))
                plt.imshow(color_image)
                plt.title('Classified Land Cover Image')
                labels = ['Forest', 'Land', 'Water', 'Urban', 'Agriculture', 'Barren', 'Unknown', 'Road']
                color_map = [
                    [0, 255, 0],       # Green (forest)
                    [255, 255, 0],     # Yellow (land)
                    [0, 0, 255],       # Blue (water)
                    [128, 128, 128],   # Gray (urban)
                    [0, 255, 255],     # Cyan (agriculture)
                    [255, 0, 0],       # Red (barren)
                    [255, 255, 255],   # White (unknown)
                    [128, 64, 0]       # Brown (road)
                ]
                handles = [plt.Rectangle((0, 0), 1, 1, color=np.array(color) / 255.0) for color in color_map]
                plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.savefig(file_path, bbox_inches='tight')
                messagebox.showinfo("Success", f"Classified image saved as {file_path}")
        else:
            messagebox.showerror("Error", "No classified image to save.")

    def save_classified_image_tif(self):
        global classified_image, image_path
        if classified_image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".tif", filetypes=[("TIF files", "*.tif"), ("All files", "*.*")])
            if file_path:
                original_image = gdal.Open(image_path)
                geo_transform = original_image.GetGeoTransform()
                projection = original_image.GetProjection()
                band_count = original_image.RasterCount
                data_type = original_image.GetRasterBand(1).DataType

                driver = gdal.GetDriverByName("GTiff")
                outdata = driver.Create(file_path, classified_image.shape[1], classified_image.shape[0], 1, data_type)
                outdata.SetGeoTransform(geo_transform)  # Set geotransform
                outdata.SetProjection(projection)  # Set projection
                outdata.GetRasterBand(1).WriteArray(classified_image)
                outdata.FlushCache()  # Save to disk
                outdata = None  # Close the file
                original_image = None  # Close the original image

                messagebox.showinfo("Success", f"Classified image saved as {file_path}")
        else:
            messagebox.showerror("Error", "No classified image to save.")

    def start_processing(self):
        threading.Thread(target=self.process_files).start()

    def process_files(self):
        global image_path, gcp_path, color_image, classified_image
        progress_text.set("Loading UAV image...")
        self.frame.update_idletasks()

        if not image_path:
            messagebox.showerror("Error", "Please select a UAV image file.")
            return

        # Load the UAV image using GDAL
        image = gdal.Open(image_path)
        image_data = image.ReadAsArray().transpose(1, 2, 0)  # Assuming the image has multiple bands
        rows, cols, bands = image_data.shape

        if rows * cols > 10000 * 10000:  # Large image threshold, adjust as needed
            progress_text.set("Large image detected, processing without displaying...")
            self.frame.update_idletasks()
        else:
            progress_text.set("Displaying UAV image...")
            self.frame.update_idletasks()
            self.display_image(image_data[:, :, :3], 'UAV Image')

        self.progress_bar["maximum"] = rows

        if self.gcp_path:
            progress_text.set("Loading GCP data...")
            self.frame.update_idletasks()

            # Load the ground control points (GCP) CSV file
            gcp_data = pd.read_csv(self.gcp_path)
            features = gcp_data[['x', 'y']]
            labels = gcp_data['land_cover_type']

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

            # Initialize the Random Forest classifier
            rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            progress_text.set("Training Random Forest classifier...")
            self.frame.update_idletasks()

            rf_classifier.fit(X_train, y_train)
            progress_text.set("Predicting land cover classes...")
            self.frame.update_idletasks()

            chunk_size = 5000  # Adjust chunk size based on available memory
            classified_image = np.zeros((rows, cols), dtype=np.int32)

            for i in range(0, rows, chunk_size):
                for j in range(0, cols, chunk_size):
                    progress_text.set(f"Processing chunk ({i}, {j})...")
                    self.frame.update_idletasks()
                    chunk = image_data[i:i+chunk_size, j:j+chunk_size]
                    chunk_reshaped = chunk.reshape((-1, bands))
                    chunk_predicted = rf_classifier.predict(chunk_reshaped)
                    classified_image[i:i+chunk_size, j:j+chunk_size] = chunk_predicted.reshape(chunk.shape[0], chunk.shape[1])
                self.progress_bar["value"] = i + chunk_size
                self.frame.update_idletasks()

            # Generate the truth table
            truth_table_gcp = gcp_data.sample(n=10, random_state=42)
            truth_table = pd.DataFrame({
                'x': truth_table_gcp['x'],
                'y': truth_table_gcp['y'],
                'True Label': truth_table_gcp['land_cover_type'],
                'Predicted Label': rf_classifier.predict(truth_table_gcp[['x', 'y']])
            })
            print(truth_table)
        else:
            progress_text.set("Performing unsupervised classification...")
            self.frame.update_idletasks()

            chunk_size = 5000  # Adjust chunk size based on available memory
            classified_image = np.zeros((rows, cols),dtype=np.int32)
        for i in range(0, rows, chunk_size):
            for j in range(0, cols, chunk_size):
                progress_text.set(f"Processing chunk ({i}, {j})...")
                self.frame.update_idletasks()
                chunk = image_data[i:i+chunk_size, j:j+chunk_size]
                chunk_reshaped = chunk.reshape((-1, bands))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=ConvergenceWarning)
                    kmeans = KMeans(n_clusters=8, random_state=42).fit(chunk_reshaped)
                chunk_predicted = kmeans.labels_
                classified_image[i:i+chunk_size, j:j+chunk_size] = chunk_predicted.reshape(chunk.shape[0], chunk.shape[1])
            self.progress_bar["value"] = i + chunk_size
            self.frame.update_idletasks()

    # Define a color map for visualization
            # Define a color map for visualization
        color_map = [
            [0, 255, 0],       # Green (forest)
            [255, 255, 0],     # Yellow (land)
            [0, 0, 255],       # Blue (water)
            [128, 128, 128],   # Gray (urban)
            [0, 255, 255],     # Cyan (agriculture)
            [255, 0, 0],       # Red (barren)
            [255, 255, 255],   # White (unknown)
            [128, 64, 0]       # Brown (road)
        ]

        color_image = np.zeros((rows, cols, 3), dtype=np.uint8)

        for i, color in enumerate(color_map):
            mask = (classified_image == i)
            for j in range(3):
                color_image[:, :, j][mask] = color[j]

        if rows * cols <= 10000 * 10000:  # Display only if the image is small enough
            progress_text.set("Displaying the classified image...")
            self.frame.update_idletasks()
            plt.figure(figsize=(10, 10))
            plt.imshow(color_image)
            plt.title('Classified Land Cover Image')
            labels = ['Forest', 'Land', 'Water', 'Urban', 'Agriculture', 'Barren', 'Unknown', 'Road']
            handles = [plt.Rectangle((0, 0), 1, 1, color=np.array(color) / 255.0) for color in color_map]
            plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.show()

        progress_text.set("Processing completed.")
        self.progress_bar["value"] = self.progress_bar["maximum"]
        self.btn_save_png.pack(pady=10)
        self.btn_save_tif.pack(pady=10)

    def select_image(self):
        global image_path
        image_path = filedialog.askopenfilename(title="Select UAV Image", filetypes=[("TIF files", "*.tif")])
        if image_path:
            self.lbl_image_path.config(text=image_path)

    def select_gcp(self):
        self.gcp_path = filedialog.askopenfilename(title="Select GCP CSV File", filetypes=[("CSV files", "*.csv")])
        if self.gcp_path:
            self.lbl_gcp_path.config(text=self.gcp_path)

    def toggle_gcp_option(self):
        if self.gcp_option.get() == "Yes":
            self.btn_upload_gcp.pack(pady=5)
            self.lbl_gcp_path.pack(pady=5)
        else:
            self.btn_upload_gcp.pack_forget()
            self.lbl_gcp_path.pack_forget()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Random Forest UAV Image Classification")
    app = RandomForestApp(root)
    root.mainloop()

