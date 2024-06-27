import tkinter as tk
from tkinter import filedialog, messagebox
from osgeo import gdal
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import ConvergenceWarning

class SVMApp:
    def __init__(self, parent):
        self.parent = parent
        self.frame = tk.Frame(self.parent)
        self.frame.pack(fill=tk.BOTH, expand=True)
        self.image_path = None
        self.gcp_path = None
        self.progress_text = tk.StringVar()
        self.color_image = None
        self.classified_image = None

        self.create_widgets()

    def create_widgets(self):
        tk.Button(self.frame, text="Upload UAV Image", command=self.select_image).pack(pady=10)
        self.lbl_image_path = tk.Label(self.frame, text="No file selected")
        self.lbl_image_path.pack(pady=5)

        tk.Label(self.frame, text="Do you want to upload Ground Control Points (GCP)?").pack(pady=5)
        self.gcp_option = tk.StringVar(value="No")
        tk.Radiobutton(self.frame, text="Yes", variable=self.gcp_option, value="Yes", command=self.toggle_gcp_option).pack()
        tk.Radiobutton(self.frame, text="No", variable=self.gcp_option, value="No", command=self.toggle_gcp_option).pack()

        self.btn_upload_gcp = tk.Button(self.frame, text="Upload GCP CSV File", command=self.select_gcp)
        self.lbl_gcp_path = tk.Label(self.frame, text="No file selected")

        tk.Label(self.frame, textvariable=self.progress_text).pack(pady=5)
        tk.Button(self.frame, text="Proceed", command=self.process_files).pack(pady=20)

        self.btn_save_png = tk.Button(self.frame, text="Save Classified Image as PNG", command=self.save_classified_image_png)
        self.btn_save_tif = tk.Button(self.frame, text="Save Classified Image as TIF", command=self.save_classified_image_tif)

    def select_image(self):
        self.image_path = filedialog.askopenfilename(title="Select UAV Image", filetypes=[("TIF files", "*.tif")])
        if self.image_path:
            self.lbl_image_path.config(text=self.image_path)

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

    def display_image(self, image_data, title='Image'):
        plt.figure(figsize=(10, 10))
        plt.imshow(image_data)
        plt.title(title)
        plt.show()

    def save_classified_image_png(self):
        if self.color_image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
            if file_path:
                plt.figure(figsize=(10, 10))
                plt.imshow(self.color_image)
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
        if self.classified_image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".tif", filetypes=[("TIF files", "*.tif"), ("All files", "*.*")])
            if file_path:
                original_image = gdal.Open(self.image_path)
                geo_transform = original_image.GetGeoTransform()
                projection = original_image.GetProjection()
                band_count = original_image.RasterCount
                data_type = original_image.GetRasterBand(1).DataType

                driver = gdal.GetDriverByName("GTiff")
                outdata = driver.Create(file_path, self.classified_image.shape[1], self.classified_image.shape[0], 1, data_type)
                outdata.SetGeoTransform(geo_transform)
                outdata.SetProjection(projection)
                outdata.GetRasterBand(1).WriteArray(self.classified_image)
                outdata.FlushCache()
                outdata = None
                original_image = None

                messagebox.showinfo("Success", f"Classified image saved as {file_path}")
        else:
            messagebox.showerror("Error", "No classified image to save.")

    def calculate_area(self, classified_image, geo_transform):
        pixel_area = geo_transform[1] * abs(geo_transform[5])
        unique_classes, counts = np.unique(classified_image, return_counts=True)
        areas = counts * pixel_area
        class_labels = ['Forest', 'Land', 'Water', 'Urban', 'Agriculture', 'Barren', 'Unknown', 'Road']
        class_areas = {class_labels[int(k)]: v for k, v in zip(unique_classes, areas)}
        total_area = sum(areas)
        return class_areas, total_area

    def process_files(self):
        self.progress_text.set("Loading UAV image...")

        if not self.image_path:
            messagebox.showerror("Error", "Please select a UAV image file.")
            return

        image = gdal.Open(self.image_path)
        image_data = image.ReadAsArray().transpose(1, 2, 0)
        geo_transform = image.GetGeoTransform()

        rows, cols, bands = image_data.shape

        if rows * cols > 10000 * 10000:
            self.progress_text.set("Large image detected, processing without displaying...")
            self.parent.update_idletasks()
        else:
            self.progress_text.set("Displaying UAV image...")
            self.parent.update_idletasks()
            self.display_image(image_data[:, :, :3], 'UAV Image')

        if self.gcp_path:
            self.progress_text.set("Loading GCP data...")
            self.parent.update_idletasks()

            gcp_data = pd.read_csv(self.gcp_path)
            features = gcp_data[['x', 'y']]
            labels = gcp_data['land_cover_type']

            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
            svm_classifier = SVC(kernel='linear', random_state=42)

            self.progress_text.set("Training SVM classifier...")
            self.parent.update_idletasks()

            svm_classifier.fit(X_train, y_train)

            self.progress_text.set("Performing SVM classification...")
            self.parent.update_idletasks()

            chunk_size = 5000
            self.classified_image = np.zeros((rows, cols), dtype=np.int32)

            for i in range(0, rows, chunk_size):
                for j in range(0, cols, chunk_size):
                    self.progress_text.set(f"Processing chunk ({i}, {j})...")
                    self.parent.update_idletasks()
                    chunk = image_data[i:i+chunk_size, j:j+chunk_size]
                    chunk_reshaped = chunk.reshape((-1, bands))
                    chunk_predicted = svm_classifier.predict(chunk_reshaped)
                    self.classified_image[i:i+chunk_size, j:j+chunk_size] = chunk_predicted.reshape(chunk.shape[0], chunk.shape[1])

            truth_table_gcp = gcp_data.sample(n=10, random_state=42)
            truth_table = pd.DataFrame({
                'x': truth_table_gcp['x'],
                'y': truth_table_gcp['y'],
                'True Label': truth_table_gcp['land_cover_type'],
                'Predicted Label': svm_classifier.predict(truth_table_gcp[['x', 'y']])
            })
            print(truth_table)
        else:
            self.progress_text.set("Performing unsupervised classification...")
            self.parent.update_idletasks()

            chunk_size = 5000
            self.classified_image = np.zeros((rows, cols), dtype=np.int32)

            for i in range(0, rows, chunk_size):
                for j in range(0, cols, chunk_size):
                    self.progress_text.set(f"Processing chunk ({i}, {j})...")
                    self.parent.update_idletasks()
                    chunk = image_data[i:i+chunk_size, j:j+chunk_size]
                    chunk_reshaped = chunk.reshape((-1, bands))
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=ConvergenceWarning)
                        kmeans = KMeans(n_clusters=8, random_state=42).fit(chunk_reshaped)
                    chunk_predicted = kmeans.labels_
                    self.classified_image[i:i+chunk_size, j:j+chunk_size] = chunk_predicted.reshape(chunk.shape[0], chunk.shape[1])

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

        self.color_image = np.zeros((rows, cols, 3), dtype=np.uint8)

        for i, color in enumerate(color_map):
            mask = (self.classified_image == i)
            for j in range(3):
                self.color_image[:, :, j][mask] = color[j]

        if rows * cols <= 10000 * 10000:
            self.progress_text.set("Displaying the classified image...")
            self.parent.update_idletasks()

            plt.figure(figsize=(10, 10))
            plt.imshow(self.color_image)
            plt.title('Classified Land Cover Image')
            labels = ['Forest', 'Land', 'Water', 'Urban', 'Agriculture', 'Barren', 'Unknown', 'Road']
            handles = [plt.Rectangle((0, 0), 1, 1, color=np.array(color) / 255.0) for color in color_map]
            plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.show()

        self.progress_text.set("Calculating area covered by each class...")
        self.parent.update_idletasks()

        class_areas, total_area = self.calculate_area(self.classified_image, geo_transform)
        area_text = "\n".join([f"{k}: {v:.2f} square units" for k, v in class_areas.items()])
        area_text += f"\nTotal Area: {total_area:.2f} square units"
        messagebox.showinfo("Area Covered", area_text)

        self.progress_text.set("Processing completed.")
        self.btn_save_png.pack(pady=10)
        self.btn_save_tif.pack(pady=10)