import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
from rf import RandomForestApp
from knn import KNNApp
from svm import SVMApp
from ClusteringBasedSegmentation import ClusteringBasedSegmentation
from Otsus import OtsusSegmentationApp
from Prewitt import prewittEdgeDetectionApp
from RegionBasedSegmentation import WatershedSegmentationApp
from Robert import RobertEdgeDetectionApp
from thresholding import ThresholdingApp
from gui_yoloV8 import YOLOGUI
from datetime import datetime
from geopy.geocoders import Nominatim
import requests

class Dashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("North Eastern Space Applications Center")
        
        # Make the window full screen
        self.root.attributes('-fullscreen', True)

        # Get the screen width
        screen_width = self.root.winfo_screenwidth()

        # Header frame
        header_frame = tk.Frame(self.root, bg="SlateGray4")
        header_frame.pack(fill=tk.X)

        # Add logo to the header
        logo_image = tk.PhotoImage(file='nesac-logo-web.png')  # Replace with the path to your logo image
        logo_label = tk.Label(header_frame, image=logo_image, bg="SlateGray4")
        logo_label.image = logo_image  # Keep a reference to the image to prevent garbage collection
        logo_label.grid(row=0, column=0, padx=(20, 10))

        # Title and subtitle frame to hold text labels
        text_frame = tk.Frame(header_frame, bg="SlateGray4")
        text_frame.grid(row=0, column=1, padx=(screen_width//2 - 450, 10), sticky='ew')  # Centering dynamically

        # Title label for the center
        title_label = tk.Label(text_frame, text="North Eastern Space Applications Center", font=("Arial", 24, "bold"), bg="SlateGray4", fg="white")
        title_label.pack(pady=(20, 10))  # Adding padding for spacing

        # Subtitle
        subtitle_label = tk.Label(text_frame, text="Government of India, Department of Space", font=("Arial", 10), bg="SlateGray4", fg="white")
        subtitle_label.pack(pady=(0, 20))
        subtitle_label = tk.Label(text_frame, text="Under the supervision of Scientist 'SD' Pradesh Jena", font=("Arial", 14), bg="SlateGray4", fg="white")
        subtitle_label.pack(pady=(0, 20))

        footer_frame = tk.Frame(self.root, bg="SlateGray4")
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM)

        footer_label = tk.Label(footer_frame, text="Developed By: Ashish Patel", font=("Arial", 14), bg="SlateGray4", fg="white")
        footer_label.pack(pady=10)

        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel for buttons
        left_panel = tk.Frame(main_frame, bg="gray", width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y)

        # Canvas for displaying images and results
        self.right_panel = tk.Frame(main_frame, bg="white")
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.right_panel, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Clock label
        self.clock_label = tk.Label(self.right_panel, font=("Arial", 20), bg="white")
        self.clock_label.pack(pady=10)

        # Location label
        self.location_label = tk.Label(self.right_panel, font=("Arial", 14), bg="white")
        self.location_label.pack(pady=10)

        self.update_clock()
        self.update_location()

        # Define button hover effects
        def on_enter(e):
            e.widget['background'] = 'lightblue'

        def on_leave(e):
            e.widget['background'] = e.widget.original_bg

        # Define LCLU classification algorithms and their respective commands
        def run_deep_learning():
            messagebox.showinfo("Algorithm Selected", "Deep Learning Algorithm")

        def run_neural_network():
            messagebox.showinfo("Algorithm Selected", "Neural Network Algorithm")

        def load_image():
            file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.tif *.tiff")])
            if file_path:
                img = Image.open(file_path)
                tk_img = ImageTk.PhotoImage(img)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
                self.canvas.image = tk_img  # Keep a reference to the image

        def exit_app():
            self.root.quit()

        # Adding buttons to left panel with rounded corners
        btn_load_image = tk.Button(left_panel, text="Load Image", command=load_image, width=25, height=2, bg="LightBlue1", activebackground="gray")
        btn_load_image.pack(pady=10)
        btn_load_image.original_bg = btn_load_image['background']
        btn_load_image.bind("<Enter>", on_enter)
        btn_load_image.bind("<Leave>", on_leave)

        tk.Label(left_panel, text="LCLU Classification Algorithms", font=("Arial", 14, "bold"), bg="gray").pack(pady=10)

        btn_deep_learning = tk.Button(left_panel, text="Deep Learning", command=self.open_YOLOGUI, width=25, height=2, bg="RoyalBlue1", activebackground="gray")
        btn_deep_learning.pack(pady=5)
        btn_deep_learning.original_bg = btn_deep_learning['background']
        btn_deep_learning.bind("<Enter>", on_enter)
        btn_deep_learning.bind("<Leave>", on_leave)

        btn_random_forest = tk.Button(left_panel, text="Random Forest", command=self.open_random_forest, width=25, height=2, bg="MediumSeaGreen", activebackground="gray")
        btn_random_forest.pack(pady=5)
        btn_random_forest.original_bg = btn_random_forest['background']
        btn_random_forest.bind("<Enter>", on_enter)
        btn_random_forest.bind("<Leave>", on_leave)

        btn_knn = tk.Button(left_panel, text="KNN", command=self.open_knn, width=25, height=2, bg="Coral", activebackground="gray")
        btn_knn.pack(pady=5)
        btn_knn.original_bg = btn_knn['background']
        btn_knn.bind("<Enter>", on_enter)
        btn_knn.bind("<Leave>", on_leave)

        btn_svm = tk.Button(left_panel, text="SVM", command=self.open_SVM, width=25, height=2, bg="MediumOrchid", activebackground="gray")
        btn_svm.pack(pady=5)
        btn_svm.original_bg = btn_svm['background']
        btn_svm.bind("<Enter>", on_enter)
        btn_svm.bind("<Leave>", on_leave)

        tk.Label(left_panel, text="Segmentation Algorithms", font=("Arial", 14, "bold"), bg="gray").pack(pady=10)

        btn_clustering = tk.Button(left_panel, text="Clustering based segmentation", command=self.open_clusteringBasedSegementation, width=25, height=2, bg="LightSalmon", activebackground="gray")
        btn_clustering.pack(pady=5)
        btn_clustering.original_bg = btn_clustering['background']
        btn_clustering.bind("<Enter>", on_enter)
        btn_clustering.bind("<Leave>", on_leave)

        btn_neural_network = tk.Button(left_panel, text="Neural Network", command=run_neural_network, width=25, height=2, bg="PaleGreen", activebackground="gray")
        btn_neural_network.pack(pady=5)
        btn_neural_network.original_bg = btn_neural_network['background']
        btn_neural_network.bind("<Enter>", on_enter)
        btn_neural_network.bind("<Leave>", on_leave)

        btn_otsu_method = tk.Button(left_panel, text="Otsu's Method", command=self.open_OtsusSegmentationApp, width=25, height=2, bg="Khaki", activebackground="gray")
        btn_otsu_method.pack(pady=5)
        btn_otsu_method.original_bg = btn_otsu_method['background']
        btn_otsu_method.bind("<Enter>", on_enter)
        btn_otsu_method.bind("<Leave>", on_leave)

        btn_prewitt_operator = tk.Button(left_panel, text="Prewitt Operator", command=self.open_prewittEdgeDetectionApp, width=25, height=2, bg="SkyBlue1", activebackground="gray")
        btn_prewitt_operator.pack(pady=5)
        btn_prewitt_operator.original_bg = btn_prewitt_operator['background']
        btn_prewitt_operator.bind("<Enter>", on_enter)
        btn_prewitt_operator.bind("<Leave>", on_leave)

        btn_region_based = tk.Button(left_panel, text="Region Based", command=self.open_WatershedSegmentationApp, width=25, height=2, bg="Thistle", activebackground="gray")
        btn_region_based.pack(pady=5)
        btn_region_based.original_bg = btn_region_based['background']
        btn_region_based.bind("<Enter>", on_enter)
        btn_region_based.bind("<Leave>", on_leave)

        btn_robert_operator = tk.Button(left_panel, text="Robert Operator", command=self.open_RobertEdgeDetectionApp, width=25, height=2, bg="Tomato", activebackground="gray")
        btn_robert_operator.pack(pady=5)
        btn_robert_operator.original_bg = btn_robert_operator['background']
        btn_robert_operator.bind("<Enter>", on_enter)
        btn_robert_operator.bind("<Leave>", on_leave)

        btn_thresholding = tk.Button(left_panel, text="Thresholding", command=self.open_ThresholdingApp, width=25, height=2, bg="SlateBlue1", activebackground="gray")
        btn_thresholding.pack(pady=5)
        btn_thresholding.original_bg = btn_thresholding['background']
        btn_thresholding.bind("<Enter>", on_enter)
        btn_thresholding.bind("<Leave>", on_leave)

        btn_exit = tk.Button(left_panel, text="Exit", command=exit_app, width=25, height=2, bg="red", activebackground="gray")
        btn_exit.pack(pady=10)
        btn_exit.original_bg = btn_exit['background']
        btn_exit.bind("<Enter>", on_enter)
        btn_exit.bind("<Leave>", on_leave)

    def clear_right_panel(self):
        for widget in self.right_panel.winfo_children():
            if widget not in (self.clock_label, self.location_label):
                widget.destroy()

    def open_random_forest(self):
        self.clear_right_panel()
        RandomForestApp(self.right_panel)

    def open_knn(self):
        self.clear_right_panel()
        KNNApp(self.right_panel)

    def open_SVM(self):
        self.clear_right_panel()
        SVMApp(self.right_panel)

    def open_YOLOGUI(self):
        self.clear_right_panel()
        YOLOGUI(self.right_panel)

    def open_clusteringBasedSegementation(self):
        self.clear_right_panel()
        ClusteringBasedSegmentation(self.right_panel)

    def open_OtsusSegmentationApp(self):
        self.clear_right_panel()
        OtsusSegmentationApp(self.right_panel)

    def open_prewittEdgeDetectionApp(self):
        self.clear_right_panel()
        prewittEdgeDetectionApp(self.right_panel)

    def open_WatershedSegmentationApp(self):
        self.clear_right_panel()
        WatershedSegmentationApp(self.right_panel)

    def open_RobertEdgeDetectionApp(self):
        self.clear_right_panel()
        RobertEdgeDetectionApp(self.right_panel)

    def open_ThresholdingApp(self):
        self.clear_right_panel()
        ThresholdingApp(self.right_panel)

    def update_clock(self):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        self.clock_label.config(text="Current Time: " + current_time)
        self.root.after(1000, self.update_clock)

    def update_location(self):
        try:
            # Fetch user's public IP address
            ip = requests.get('https://api64.ipify.org').text
            # Fetch location information based on IP
            response = requests.get(f'https://ipinfo.io/{ip}/json')
            data = response.json()
            if response.status_code == 200 and 'loc' in data:
                location = data['loc'].split(',')
                latitude = location[0]
                longitude = location[1]
                city = data.get('city', 'Unknown City')
                region = data.get('region', 'Unknown Region')
                country = data.get('country', 'Unknown Country')
                self.location_label.config(text=f"Location: {city}, {region}, {country} (Lat: {latitude}, Long: {longitude})")
            else:
                self.location_label.config(text="Location: Unknown")
        except Exception as e:
            self.location_label.config(text=f"Location: Error {e}")

if __name__ == "__main__":
    root = tk.Tk()
    dashboard = Dashboard(root)
    root.mainloop()
