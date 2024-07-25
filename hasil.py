import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Definisikan transformasi yang sama seperti saat pelatihan
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load kembali model
model = models.resnet50()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Sesuaikan dengan jumlah kelas yang Anda gunakan saat melatih

model.load_state_dict(torch.load('resnet50_signature_model.pth'))
model.eval()  # Set model ke mode evaluasi

# Fungsi untuk memprediksi kelas dari gambar
def predict_image(image_path, model, transform):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Tambahkan dimensi batch
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()
        return class_idx

# Path ke gambar yang ingin Anda uji
image_path = 'dataset/personality/test/ordinary/2019230142.jpg'  # Ganti dengan path gambar uji Anda

# Prediksi kelas dari gambar
class_idx = predict_image(image_path, model, data_transforms)

# Daftar nama kelas yang sesuai dengan indeks kelas
class_names = ['ordinary', 'strong']  # Sesuaikan dengan urutan kelas yang digunakan saat melatih

# Tampilkan hasil prediksi
predicted_class = class_names[class_idx]
print(f'Hasil prediksi: {predicted_class}')

# Tampilkan gambar uji
image = Image.open(image_path)
plt.imshow(image)
plt.title(f'Prediksi: {predicted_class}')
plt.axis('off')
plt.show()
