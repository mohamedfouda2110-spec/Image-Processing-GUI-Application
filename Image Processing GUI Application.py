import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage import io, color, filters, exposure, morphology, img_as_ubyte

class ImageProcessingProject:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing Project - Tanta University")
        self.root.geometry("1200x800")
        self.root.configure(bg="#2c3e50")

        # تصحيح المسار باستخدام حرف r و __file__
        self.default_path = r"D:/Desktop-D/New folder/mo.jpg"
        
        self.original_img = None
        self.processed_img = None
        
        self.create_widgets()
        self.auto_load_image()

    def create_widgets(self):
        control_panel = tk.Frame(self.root, bg="#34495e", width=250)
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        tk.Label(control_panel, text="قائمة العمليات", fg="white", bg="#34495e", font=("Arial", 14, "bold")).pack(pady=10)

        actions = [
            ("Gray Scale (2)", self.to_gray),
            ("Negative (3)", self.make_negative),
            ("Log Transform (3)", self.log_transform),
            ("Gamma (0.5) (3)", lambda: self.gamma_correction(0.5)),
            ("Histogram (4)", self.hist_equal),
            ("Median Filter (5)", self.median_blur),
            ("Sobel Edges (5)", self.apply_sobel),
            ("Morph. Opening (6)", self.apply_opening),
            ("Reset Image", self.auto_load_image)
        ]

        for text, func in actions:
             tk.Button(control_panel, text=text, command=func, bg="#ecf0f1", width=20).pack(fill=tk.X, pady=2, padx=10)

        self.display_frame = tk.Frame(self.root, bg="gray")
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

    def auto_load_image(self):
        try:
            # التحميل والتأكد من الصيغة
            self.original_img = io.imread(self.default_path)
            if self.original_img.dtype == np.uint8:
                self.original_img = self.original_img.astype(float) / 255.0
            
            self.processed_img = np.copy(self.original_img)
            self.show_image()
        except Exception as e:
            messagebox.showerror("Error", f"لم يتم العثور على الصورة في المسار المحدد:\n{self.default_path}")

    def show_image(self):
        for widget in self.display_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(8, 6))
        if len(self.processed_img.shape) == 2:
            ax.imshow(self.processed_img, cmap='gray')
        else:
            ax.imshow(self.processed_img)
        
        ax.axis('off')
        canvas = FigureCanvasTkAgg(fig, master=self.display_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # --- العمليات المجمعة من السكاشن ---

    def to_gray(self):
        if len(self.processed_img.shape) == 3:
            self.processed_img = color.rgb2gray(self.processed_img)
            self.show_image()

    def make_negative(self):
        self.processed_img = 1.0 - self.processed_img
        self.show_image()

    def log_transform(self):
        c = 1 / np.log(1 + np.max(self.processed_img))
        self.processed_img = c * (np.log(1 + self.processed_img))
        self.show_image()

    def gamma_correction(self, g):
        self.processed_img = exposure.adjust_gamma(self.processed_img, gamma=g)
        self.show_image()

    def hist_equal(self):
        self.processed_img = exposure.equalize_hist(self.processed_img)
        self.show_image()

    def median_blur(self): # 
        if self.processed_img is not None:
            # تحويل الصورة لصيغة ubyte (0-255) ليعمل الميدين فلتر بشكل صحيح
            temp_img = img_as_ubyte(exposure.rescale_intensity(self.processed_img))
            self.processed_img = filters.median(temp_img)
            self.show_image()

    def apply_sobel(self):
        self.to_gray() 
        self.processed_img = filters.sobel(self.processed_img)
        self.show_image()

    def apply_opening(self):
        self.to_gray()
        thresh = filters.threshold_otsu(self.processed_img)
        binary = self.processed_img > thresh
        self.processed_img = morphology.binary_opening(binary, morphology.disk(3))
        self.show_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingProject(root)
    root.mainloop()