import math
import tkinter as tk
from tkinter import Label, Button, filedialog, messagebox, Canvas, Frame, Scrollbar
from PIL import Image, ImageTk

class ClusteringBasedSegmentation:
    def __init__(self, parent):
        self.parent = parent
        self.frame = Frame(self.parent, bg="white")
        self.frame.pack(fill=tk.BOTH, expand=True)
        
        self.input_image_path = None
        self.input_image = None
        self.cluster_centers = None
        self.clusters = None

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

        self.btn_upload_image = Button(self.scrollable_frame, text="Upload Image", command=self.select_image)
        self.btn_upload_image.pack(pady=10)

        self.lbl_image_path = Label(self.scrollable_frame, text="No file selected")
        self.lbl_image_path.pack(pady=5)

        self.lbl_input_image = Label(self.scrollable_frame)
        self.lbl_input_image.pack(pady=10)

        self.btn_perform_kmeans = Button(self.scrollable_frame, text="Perform K-means Clustering", command=self.perform_kmeans)
        self.btn_perform_kmeans.pack(pady=10)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

    def select_image(self):
        self.input_image_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.tif;*.bmp")])
        if self.input_image_path:
            self.lbl_image_path.config(text=f"Selected Image: {self.input_image_path}")
            try:
                pil_image = Image.open(self.input_image_path)
                self.input_image = ImageTk.PhotoImage(pil_image)
                self.lbl_input_image.config(image=self.input_image)
                self.lbl_input_image.image = self.input_image  # Keep a reference to prevent garbage collection
                messagebox.showinfo("Image Selected", f"Selected image: {self.input_image_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def perform_kmeans(self):
        if self.input_image is None:
            messagebox.showerror("Error", "Please select an image first.")
            return

        try:
            # Dummy implementation of K-means clustering
            n = 5  # Number of points
            a = 3  # Number of clusters

            # Dummy data for clustering (replace with your logic)
            l1 = [1, 2, 3, 4, 5]  # Example x-coordinates
            l2 = [1, 1, 2, 2, 3]  # Example y-coordinates
            p = list(zip(l1, l2))  # List of tuples [(1, 1), (2, 1), (3, 2), (4, 2), (5, 3)]

            k = []
            b = []

            for i in range(a):
                m = list(p[i])
                k.append(m)
                b.append(m)

            y = []

            while True:
                for j in range(a):
                    x = []
                    for i in range(n):
                        q = math.pow(p[i][0] - k[j][0], 2)
                        w = math.pow(p[i][1] - k[j][1], 2)
                        t = math.sqrt(q + w)
                        x.append(t)
                        h = list(x)
                    y.append(h)

                g = []
                for i in range(a):
                    g.append([])

                for i in range(n):
                    min1 = y[0][i]
                    for j in range(1, a):
                        if min1 > y[j][i]:
                            g[j].append(i)
                        else:
                            g[0].append(i)

                k.clear()
                for i in range(a):
                    j = 0
                    s1 = 0.0
                    s2 = 0.0
                    while j < len(g[i]):
                        e = g[i][j]
                        s1 = s1 + l1[e]
                        s2 = s2 + l2[e]
                        j = j + 1
                    c1 = s1 / len(g[i])
                    c2 = s2 / len(g[i])
                    k.append([c1, c2])

                x.clear()
                y.clear()
                g.clear()

                if b == k:
                    break
                else:
                    b.clear()
                    b = k

            # After clustering, update global variables or display results as needed
            self.cluster_centers = k
            self.clusters = g  # Update clusters if necessary

            # Display final centroids and clusters in GUI or print them
            print("Final centroids are = ", k)
            print("Final clusters are = ", g)

        except Exception as e:
            messagebox.showerror("Error", f"Error during K-means clustering: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ClusteringBasedSegmentation(root)
    root.mainloop()
