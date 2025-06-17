import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Đọc dữ liệu từ file CSV
# Giả sử file 'Crop_production_in_India_ok.csv' đã được cung cấp
# Nếu cần, thay thế đường dẫn file phù hợp
data = pd.read_csv('Crop_production_in_India_ok.csv')

# Làm sạch dữ liệu: Loại bỏ các giá trị sản lượng âm
data = data[data['Production'] >= 0]

# Chuyển đổi các cột số sang kiểu float
numeric_cols = ['Area', 'Temperature', 'Humidity', 'Wind_Speed', 'Production']
data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Thiết lập kiểu biểu đồ
# plt.style.use('seaborn')  # Sử dụng kiểu seaborn cho giao diện đẹp

# 1. Biểu đồ cột: Sản lượng trung bình theo cây trồng và mùa vụ
plt.figure(figsize=(12, 6))
avg_production = data.groupby(['Season', 'Crop'])['Production'].mean().unstack()
avg_production.plot(kind='bar', stacked=False, colormap='Set2')
plt.title('Average Production by Crop and Season', fontsize=14)
plt.xlabel('Season', fontsize=12)
plt.ylabel('Average Production (tons)', fontsize=12)
plt.legend(title='Crop')
plt.tight_layout()
plt.savefig('bar_chart.png')
plt.close()

# 2. Biểu đồ đường: Xu hướng sản lượng theo thời gian
plt.figure(figsize=(12, 6))
yearly_production = data.groupby(['Crop_Year', 'Crop'])['Production'].mean().unstack()
yearly_production.plot(kind='line', marker='o', colormap='Set1')
plt.title('Production Trends by Crop (1990–2024)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Average Production (tons)', fontsize=12)
plt.legend(title='Crop')
plt.tight_layout()
plt.savefig('line_chart.png')
plt.close()

# 3. Biểu đồ hộp: Phân phối sản lượng theo cây trồng
plt.figure(figsize=(10, 6))
sns.boxplot(x='Crop', y='Production', data=data, palette='Set3')
plt.title('Production Distribution by Crop', fontsize=14)
plt.xlabel('Crop', fontsize=12)
plt.ylabel('Production (tons)', fontsize=12)
plt.tight_layout()
plt.savefig('box_plot.png')
plt.close()

# 4. Biểu đồ phân tán: Sản lượng vs Nhiệt độ và Độ ẩm

# Các biến đầu vào cần phân tích
features = ['Area', 'Temperature', 'Humidity', 'Wind_Speed']

# Thiết lập kích thước hình vẽ
plt.figure(figsize=(10, 6))

# Vẽ từng biểu đồ scatter có kèm đường hồi quy
for i, feature in enumerate(features, 1):
    plt.subplot(2, 2, i)
    sns.regplot(data=data, x=feature, y='Production',
                scatter_kws={'alpha': 0.5 ,'color': 'orange'},
                line_kws={'color': 'red'})
    plt.title(f'Production vs {feature}')
    plt.xlabel(feature)
    plt.ylabel('Production')

plt.tight_layout()
plt.show()

# 5 biểu đồ histogram

# Đọc dữ liệu
df = pd.read_csv("Crop_production_in_India_ok.csv")

# Các biến liên tục để vẽ histogram
features = ['Production','Area', 'Temperature', 'Humidity', 'Wind_Speed']

# Cấu hình kiểu seaborn
sns.set(style="whitegrid")

# Vẽ histogram cho từng biến
plt.figure(figsize=(8, 6))

for i, feature in enumerate(features, 1):
    plt.subplot(3, 2, i)
    sns.histplot(df[feature], kde=True, color='#5DADE2', bins=30)
    plt.title(f'Distribution of {feature}', fontsize=14)
    plt.xlabel(feature)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
# 6 biểu đồ tròn
# Tính tổng sản lượng theo từng loại cây trồng
crop_production = data.groupby('Crop')['Production'].sum().sort_values(ascending=False)

# Chọn top 8 cây trồng lớn nhất, nhóm phần còn lại vào "Others"
top_n = 8
top_crops = crop_production[:top_n]
others = crop_production[top_n:].sum()
top_crops['Others'] = others

# Vẽ biểu đồ tròn
plt.figure(figsize=(10, 10))
plt.pie(
    top_crops,
    labels=top_crops.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=plt.cm.Paired.colors  # bảng màu đẹp
)
plt.title('Tỷ lệ sản lượng theo loại cây trồng (Crop)', fontsize=14)
plt.axis('equal')  # hình tròn đúng tỷ lệ
plt.show()
# 7 heatmap
# Chọn các cột số để tính tương quan
numeric_cols = ['Production', 'Area', 'Temperature', 'Humidity', 'Wind_Speed']

# Tính ma trận tương quan
corr_matrix = data[numeric_cols].corr()

# Vẽ heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_matrix,
    annot=True,        # Hiển thị giá trị số trên từng ô
    cmap='coolwarm',   # Bảng màu từ lạnh → nóng
    fmt='.2f',
    linewidths=0.5,
    square=True
)
plt.title('Biểu đồ Heatmap tương quan giữa các biến', fontsize=14)
plt.tight_layout()
plt.show()
