import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import torch
from torchvision import models
from torchvision.models import EfficientNet_V2_S_Weights
import json
import os
import random
from datetime import datetime
import threading
import numpy as np
from pathlib import Path


class ZooAssistantApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Zoo Thông Minh - Nhận Diện Động Vật")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f8ff')

        # Initialize model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_model()
        self.load_animal_data()

        # Camera state
        self.cap = None
        self.camera_thread = None
        self.camera_running = False
        self.current_frame = None

        # Create GUI
        self.create_widgets()

    def load_model(self):
        """Load the trained model"""
        weights = EfficientNet_V2_S_Weights.DEFAULT
        self.data_transforms = weights.transforms()

        # Model configuration
        path = "weights_3/Epoch10_Acc0.9537.pth"
        num_classes = 90

        # Initialize the model
        self.model = models.efficientnet_v2_s(weights=weights).to(self.device)
        self.model.classifier[1] = torch.nn.Linear(
            self.model.classifier[1].in_features, num_classes).to(self.device)

        # Load state dictionary
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

        # Load animal names
        with open('name.txt', 'r') as file:
            self.animal_names = [line.strip() for line in file.readlines()]
        self.animal_names.sort()

    def load_animal_data(self):
        """Load animal information from JSON file"""
        if os.path.exists('animal_info.json'):
            with open('animal_info.json', 'r', encoding='utf-8') as f:
                self.animal_info = json.load(f)
        else:
            messagebox.showwarning(
                "Cảnh báo", "Không tìm thấy file animal_info.json. Vui lòng tạo file này trước!")
            self.animal_info = {}

    def create_widgets(self):
        """Create the main GUI interface"""
        # Main title
        title_label = tk.Label(self.root, text="🦁 ZOO THÔNG MINH - HỆ THỐNG NHẬN DIỆN ĐỘNG VẬT 📸",
                               font=('Arial', 24, 'bold'), bg='#f0f8ff', fg='#2c3e50')
        title_label.pack(pady=20)

        # Main frame
        main_frame = tk.Frame(self.root, bg='#f0f8ff')
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)

        # Left side - Camera/Image section
        left_frame = tk.Frame(main_frame, bg='white', relief='raised', bd=2)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))

        # Camera controls
        camera_frame = tk.Frame(left_frame, bg='white')
        camera_frame.pack(pady=10)

        self.camera_btn = tk.Button(camera_frame, text="🎥 Mở Camera", command=self.toggle_camera,
                                    font=('Arial', 12, 'bold'), bg='#3498db', fg='white',
                                    padx=20, pady=8)
        self.camera_btn.pack(side='left', padx=5)

        capture_btn = tk.Button(camera_frame, text="📸 Chụp Ảnh", command=self.capture_photo,
                                font=('Arial', 12, 'bold'), bg='#e74c3c', fg='white',
                                padx=20, pady=8)
        capture_btn.pack(side='left', padx=5)

        upload_btn = tk.Button(camera_frame, text="📁 Tải Ảnh", command=self.upload_image,
                               font=('Arial', 12, 'bold'), bg='#9b59b6', fg='white',
                               padx=20, pady=8)
        upload_btn.pack(side='left', padx=5)

        # Prediction button
        predict_btn = tk.Button(camera_frame, text="🔍 Nhận Diện", command=self.predict_animal,
                                font=('Arial', 12, 'bold'), bg='#27ae60', fg='white',
                                padx=20, pady=8)
        predict_btn.pack(side='left', padx=5)

        # Camera/Image display
        self.image_label = tk.Label(left_frame, bg='white', text="Camera/Ảnh sẽ hiển thị ở đây",
                                    font=('Arial', 14), relief='sunken', bd=2)
        self.image_label.pack(pady=10, padx=10, fill='both', expand=True)

        # Result display
        result_frame = tk.Frame(left_frame, bg='white')
        result_frame.pack(pady=10, fill='x')

        tk.Label(result_frame, text="Kết quả nhận diện:",
                 font=('Arial', 14, 'bold'), bg='white').pack()
        self.result_label = tk.Label(result_frame, text="Chưa có kết quả", font=('Arial', 16),
                                     bg='white', fg='#2c3e50')
        self.result_label.pack(pady=5)

        # Right side - Animal information
        right_frame = tk.Frame(main_frame, bg='white', relief='raised', bd=2)
        right_frame.pack(side='right', fill='both', expand=True, padx=(10, 0))

        # Animal info title
        info_title = tk.Label(right_frame, text="📚 THÔNG TIN CHI TIẾT ĐỘNG VẬT",
                              font=('Arial', 18, 'bold'), bg='white', fg='#2c3e50')
        info_title.pack(pady=10)

        # Reference image frame
        ref_frame = tk.Frame(right_frame, bg='white')
        ref_frame.pack(pady=10)

        tk.Label(ref_frame, text="Ảnh tham khảo:", font=(
            'Arial', 12, 'bold'), bg='white').pack()
        self.ref_image_label = tk.Label(ref_frame, bg='white', relief='sunken', bd=2,
                                        text="Ảnh tham khảo sẽ hiển thị ở đây")
        self.ref_image_label.pack(pady=5)

        # Create scrollable text widget for animal information
        info_frame = tk.Frame(right_frame, bg='white')
        info_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Scrollbar
        scrollbar = tk.Scrollbar(info_frame)
        scrollbar.pack(side='right', fill='y')

        # Text widget with scrollbar
        self.info_text = tk.Text(info_frame, yscrollcommand=scrollbar.set, font=('Arial', 11),
                                 wrap='word', bg='#f8f9fa', relief='sunken', bd=2)
        self.info_text.pack(fill='both', expand=True)
        scrollbar.config(command=self.info_text.yview)

        # Initial info text
        self.info_text.insert(
            tk.END, "🌟 Chào mừng bạn đến với Zoo Thông Minh!\n\n")
        self.info_text.insert(tk.END, "Hệ thống này giúp bạn:\n")
        self.info_text.insert(
            tk.END, "• Chụp ảnh từ camera hoặc tải ảnh có sẵn\n")
        self.info_text.insert(tk.END, "• Nhận diện loài động vật trong ảnh\n")
        self.info_text.insert(
            tk.END, "• Hiển thị thông tin chi tiết về loài được nhận diện\n\n")
        self.info_text.insert(
            tk.END, "Hãy bắt đầu bằng cách mở camera hoặc tải ảnh lên!")
        self.info_text.config(state='disabled')

    def toggle_camera(self):
        """Toggle camera on/off"""
        if not self.camera_running:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        """Start camera capture"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Lỗi", "Không thể mở camera!")
                return

            self.camera_running = True
            self.camera_btn.config(text="📴 Đóng Camera", bg='#e74c3c')
            self.camera_thread = threading.Thread(target=self.update_camera)
            self.camera_thread.daemon = True
            self.camera_thread.start()
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi mở camera: {str(e)}")

    def stop_camera(self):
        """Stop camera capture"""
        self.camera_running = False
        if self.cap:
            self.cap.release()
        self.camera_btn.config(text="🎥 Mở Camera", bg='#3498db')

    def update_camera(self):
        """Update camera feed"""
        while self.camera_running:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                # Convert frame to display format
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (640, 480))
                image = Image.fromarray(frame)
                photo = ImageTk.PhotoImage(image)

                # Update display
                self.image_label.config(image=photo, text="")
                self.image_label.image = photo

    def capture_photo(self):
        """Capture photo from camera"""
        if self.current_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"captured_{timestamp}.jpg"
            cv2.imwrite(filename, self.current_frame)
            messagebox.showinfo("Thành công", f"Đã chụp ảnh: {filename}")

            # Display captured image
            frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (640, 480))
            image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
            self.current_image = image
        else:
            messagebox.showwarning(
                "Cảnh báo", "Chưa có ảnh để chụp. Vui lòng mở camera trước!")

    def upload_image(self):
        """Upload image from file"""
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )

        if file_path:
            try:
                image = Image.open(file_path)
                # Resize for display
                display_image = image.copy()
                display_image.thumbnail((640, 480))
                photo = ImageTk.PhotoImage(display_image)

                self.image_label.config(image=photo, text="")
                self.image_label.image = photo
                self.current_image = image
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể tải ảnh: {str(e)}")

    def predict_animal(self):
        """Predict animal from current image"""
        if hasattr(self, 'current_image'):
            try:
                # Preprocess image
                img_tensor = self.data_transforms(
                    self.current_image).to(self.device)
                img_tensor = img_tensor.unsqueeze(0)

                # Predict
                with torch.no_grad():
                    outputs = self.model(img_tensor)
                    _, predicted = torch.max(outputs, 1)
                    confidence = torch.softmax(outputs, 1)[0][predicted].item()

                animal_name = self.animal_names[predicted.item()]
                self.result_label.config(
                    text=f"{animal_name} ({confidence*100:.1f}%)")

                # Display animal information
                self.display_animal_info(animal_name)

            except Exception as e:
                messagebox.showerror("Lỗi", f"Lỗi khi nhận diện: {str(e)}")
        else:
            messagebox.showwarning(
                "Cảnh báo", "Vui lòng chụp ảnh hoặc tải ảnh lên trước!")

    def display_animal_info(self, animal_name):
        """Display detailed information about the predicted animal"""
        self.info_text.config(state='normal')
        self.info_text.delete(1.0, tk.END)

        if animal_name in self.animal_info:
            info = self.animal_info[animal_name]

            # Display reference image from assets folder
            self.display_reference_image_from_assets(animal_name)

            # Format and display information
            self.info_text.insert(
                tk.END, f"🦁 {info.get('name_vi', animal_name)}\n", 'title')
            self.info_text.insert(
                tk.END, f"({animal_name.title()})\n\n", 'subtitle')

            self.info_text.insert(tk.END, "📋 Phân loại:\n", 'header')
            self.info_text.insert(
                tk.END, f"{info.get('classification', 'Không có thông tin')}\n\n")

            self.info_text.insert(tk.END, "🏠 Môi trường sống:\n", 'header')
            self.info_text.insert(
                tk.END, f"{info.get('habitat', 'Không có thông tin')}\n\n")

            self.info_text.insert(tk.END, "🍃 Chế độ ăn:\n", 'header')
            self.info_text.insert(
                tk.END, f"{info.get('diet', 'Không có thông tin')}\n\n")

            self.info_text.insert(tk.END, "📏 Kích thước:\n", 'header')
            self.info_text.insert(
                tk.END, f"{info.get('size', 'Không có thông tin')}\n\n")

            self.info_text.insert(tk.END, "⚖️ Cân nặng:\n", 'header')
            self.info_text.insert(
                tk.END, f"{info.get('weight', 'Không có thông tin')}\n\n")

            self.info_text.insert(tk.END, "🎯 Tình trạng bảo tồn:\n", 'header')
            self.info_text.insert(
                tk.END, f"{info.get('conservation_status', 'Không có thông tin')}\n\n")

            self.info_text.insert(tk.END, "💡 Thông tin thú vị:\n", 'header')
            facts = info.get('interesting_facts', [])
            for i, fact in enumerate(facts, 1):
                self.info_text.insert(tk.END, f"{i}. {fact}\n")
        else:
            self.info_text.insert(
                tk.END, f"Không tìm thấy thông tin chi tiết cho {animal_name}")

        self.info_text.config(state='disabled')

        # Configure text tags for formatting
        self.info_text.tag_config('title', font=(
            'Arial', 16, 'bold'), foreground='#2c3e50')
        self.info_text.tag_config('subtitle', font=(
            'Arial', 12, 'italic'), foreground='#7f8c8d')
        self.info_text.tag_config('header', font=(
            'Arial', 12, 'bold'), foreground='#3498db')

    def display_reference_image_from_assets(self, animal_name):
        """Display reference image for the animal from assets folder"""
        assets_dir = Path("assets")

        # Tìm ảnh trong thư mục assets với các phần mở rộng khác nhau
        possible_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_path = None

        for ext in possible_extensions:
            potential_path = assets_dir / f"{animal_name}{ext}"
            if potential_path.exists():
                image_path = potential_path
                break

        if image_path:
            try:
                ref_image = Image.open(image_path)
                # Resize để vừa khung hiển thị
                ref_image.thumbnail((300, 300))
                ref_photo = ImageTk.PhotoImage(ref_image)

                self.ref_image_label.config(image=ref_photo, text="")
                self.ref_image_label.image = ref_photo
            except Exception as e:
                self.ref_image_label.config(
                    image="", text=f"Lỗi tải ảnh: {str(e)}")
        else:
            # Fallback: thử tìm trong thư mục gốc (animals/animals/...)
            self.display_reference_image_from_dataset(animal_name)

    def display_reference_image_from_dataset(self, animal_name):
        """Fallback: Display reference image from the original dataset"""
        # Look for reference image in animals folder
        animal_folder = f"animals/animals/{animal_name}"

        if os.path.exists(animal_folder):
            # Get random image from folder
            images = [f for f in os.listdir(
                animal_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                random_image = random.choice(images)
                image_path = os.path.join(animal_folder, random_image)

                try:
                    ref_image = Image.open(image_path)
                    ref_image.thumbnail((300, 300))
                    ref_photo = ImageTk.PhotoImage(ref_image)

                    self.ref_image_label.config(image=ref_photo, text="")
                    self.ref_image_label.image = ref_photo
                except Exception as e:
                    self.ref_image_label.config(
                        image="", text=f"Lỗi tải ảnh: {str(e)}")
            else:
                self.ref_image_label.config(
                    image="", text="Không tìm thấy ảnh tham khảo")
        else:
            self.ref_image_label.config(
                image="", text="Không tìm thấy thư mục ảnh")

    def __del__(self):
        """Cleanup when app is closed"""
        if self.cap:
            self.cap.release()


def main():
    root = tk.Tk()
    app = ZooAssistantApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: [
                  app.stop_camera(), root.destroy()])
    root.mainloop()


if __name__ == "__main__":
    main()
