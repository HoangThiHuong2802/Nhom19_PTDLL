import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import time
import numpy as np
import joblib

# === Đọc và xử lý dữ liệu ===
df = pd.read_csv("Crop_production_in_India_ok.csv")
print(f"Số lượng bản ghi ban đầu: {len(df)}")
df.dropna(inplace=True)
df = df[df['Production'] >= 0]  # Loại bỏ giá trị âm trong Production
print(f"Số lượng bản ghi sau khi làm sạch: {len(df)}")
df['Crop'] = df['Crop'].str.strip().str.title()
df['Season'] = df['Season'].str.strip().str.title()

# Mã hóa one-hot
df_encoded = pd.get_dummies(df, columns=['Crop', 'Season'], drop_first=True)
X = df_encoded.drop(columns=['Production'])
y = df_encoded['Production']

encoded_columns = X.columns  # Lưu danh sách cột đã mã hóa
print(f"Encoded columns: {encoded_columns.tolist()}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
start = time.time()
model.fit(X_train, y_train)
train_time = round(time.time() - start, 4)

# Đánh giá mô hình
train_r2 = model.score(X_train, y_train)
test_r2 = model.score(X_test, y_test)
print(f"Train R-squared: {train_r2:.4f}")
print(f"Test R-squared: {test_r2:.4f}")

# Lưu mô hình
joblib.dump(model, 'linear_regression_model.pkl')

# === Giao diện Tkinter ===
root = tk.Tk()
root.title("Ứng dụng Dự báo Sản lượng Cây trồng")
root.geometry("1000x700")

style = ttk.Style()
style.configure("TFrame", background="#f0f4f8")
style.configure("TLabel", background="#f0f4f8", font=("Segoe UI", 10))
style.configure("TButton", font=("Segoe UI", 10, "bold"))

tab_control = ttk.Notebook(root)
tabs = [ttk.Frame(tab_control) for _ in range(5)]
tab_titles = ['Nhập liệu', 'Dự báo', 'Biểu đồ', 'Phân tích', 'Tra cứu']
for tab, title in zip(tabs, tab_titles):
    tab_control.add(tab, text=title)
tab_control.pack(expand=1, fill="both")
tab1, tab2, tab3, tab4, tab5 = tabs

# === TAB 1: Nhập liệu ===
tk.Label(tab1, text="Loại cây trồng:").grid(row=0, column=0, padx=10, pady=10, sticky="e")
crop_cb = ttk.Combobox(tab1, values=sorted(df["Crop"].unique()), width=30)
crop_cb.grid(row=0, column=1, padx=10, pady=10)

tk.Label(tab1, text="Năm trồng (Crop_Year):").grid(row=1, column=0, padx=10, pady=10, sticky="e")
year_entry = ttk.Entry(tab1, width=33)
year_entry.grid(row=1, column=1, padx=10, pady=10)

tk.Label(tab1, text="Mùa vụ:").grid(row=2, column=0, padx=10, pady=10, sticky="e")
season_cb = ttk.Combobox(tab1, values=sorted(df["Season"].unique()), width=30)
season_cb.grid(row=2, column=1, padx=10, pady=10)

tk.Label(tab1, text="Diện tích (ha):").grid(row=3, column=0, padx=10, pady=10, sticky="e")
area_entry = ttk.Entry(tab1, width=33)
area_entry.grid(row=3, column=1, padx=10, pady=10)

tk.Label(tab1, text="Nhiệt độ (°C):").grid(row=4, column=0, padx=10, pady=10, sticky="e")
temp_entry = ttk.Entry(tab1, width=33)
temp_entry.grid(row=4, column=1, padx=10, pady=10)

tk.Label(tab1, text="Độ ẩm (%):").grid(row=5, column=0, padx=10, pady=10, sticky="e")
humid_entry = ttk.Entry(tab1, width=33)
humid_entry.grid(row=5, column=1, padx=10, pady=10)

tk.Label(tab1, text="Tốc độ gió (km/h):").grid(row=6, column=0, padx=10, pady=10, sticky="e")
wind_entry = ttk.Entry(tab1, width=33)
wind_entry.grid(row=6, column=1, padx=10, pady=10)

result_label = tk.Label(tab2, text="", font=("Segoe UI", 12, "bold"))
result_label.pack(pady=20)

def du_bao():
    try:
        crop_year = int(year_entry.get())
        area = float(area_entry.get())
        temp = float(temp_entry.get())
        humid = float(humid_entry.get())
        wind = float(wind_entry.get())
        crop = crop_cb.get().strip().title()
        season = season_cb.get().strip().title()

        if not crop or not season:
            raise ValueError("Chưa chọn loại cây hoặc mùa vụ")

        # Tạo DataFrame với dữ liệu người dùng
        user_df = pd.DataFrame([{
            'Crop_Year': crop_year,
            'Area': area,
            'Temperature': temp,
            'Humidity': humid,
            'Wind_Speed': wind,
            'Crop': crop,
            'Season': season
        }])

        # Lấy danh sách giá trị độc nhất từ dữ liệu huấn luyện để mã hóa
        all_seasons = df['Season'].unique()
        all_crops = df['Crop'].unique()

        # Tạo các cột one-hot dựa trên tất cả các giá trị có thể có
        for s in all_seasons[1:]:  # Bỏ qua giá trị đầu tiên do drop_first=True
            user_df[f'Season_{s}'] = (user_df['Season'] == s).astype(int)
        for c in all_crops[1:]:  # Bỏ qua giá trị đầu tiên do drop_first=True
            user_df[f'Crop_{c}'] = (user_df['Crop'] == c).astype(int)

        # Loại bỏ cột gốc sau khi mã hóa
        user_df = user_df.drop(columns=['Crop', 'Season'])

        # Đảm bảo thứ tự cột khớp với encoded_columns
        user_df_encoded = user_df.reindex(columns=encoded_columns, fill_value=0)

        print(f"User input columns: {user_df_encoded.columns.tolist()}")

        y_pred = model.predict(user_df_encoded)[0]
        result_label.config(text=f"Sản lượng dự báo: {max(y_pred, 0):.2f} tấn")
    except Exception as e:
        messagebox.showerror("Lỗi", f"Lỗi dữ liệu đầu vào: {e}")

def xoa():
    area_entry.delete(0, tk.END)
    temp_entry.delete(0, tk.END)
    humid_entry.delete(0, tk.END)
    wind_entry.delete(0, tk.END)
    year_entry.delete(0, tk.END)
    crop_cb.set("")
    season_cb.set("")
    result_label.config(text="")

tk.Button(tab1, text="Dự báo sản lượng", command=du_bao, bg="#4CAF50", fg="white").grid(row=7, column=0, padx=10, pady=20)
tk.Button(tab1, text="Xoá", command=xoa, bg="#f44336", fg="white").grid(row=7, column=1, padx=10, pady=20)

# === TAB 2: Dự báo ===
tk.Label(tab2, text="Mô hình: Hồi quy tuyến tính", font=("Segoe UI", 11)).pack(pady=10)
tk.Label(tab2, text=f"Thời gian huấn luyện: {train_time} giây", font=("Segoe UI", 11)).pack(pady=10)
tk.Label(tab2, text=f"Train R-squared: {train_r2:.4f}", font=("Segoe UI", 11)).pack(pady=10)
tk.Label(tab2, text=f"Test R-squared: {test_r2:.4f}", font=("Segoe UI", 11)).pack(pady=10)

# === TAB 3: Biểu đồ ===
chart_frame = tk.Frame(tab3)
chart_frame.pack()

def ve_bieu_do(loai):
    for widget in chart_frame.winfo_children():
        widget.destroy()
    fig = plt.Figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    if loai == "Sản lượng theo năm":
        avg = df.groupby("Crop_Year")["Production"].mean()
        avg.plot(kind="line", ax=ax)
        ax.set_title("Sản lượng trung bình theo năm")
    elif loai == "Top 3 cây trồng":
        top = df.groupby("Crop")["Production"].sum().sort_values(ascending=False).head(3)
        top.plot(kind="bar", ax=ax)
        ax.set_title("Top 3 cây trồng có sản lượng cao nhất")
    elif loai == "Diện tích vs Sản lượng":
        ax.scatter(df["Area"], df["Production"], alpha=0.3)
        ax.set_title("Diện tích vs Sản lượng")
    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

chart_options = ["Sản lượng theo năm", "Top 3 cây trồng", "Diện tích vs Sản lượng"]
chart_cb = ttk.Combobox(tab3, values=chart_options, width=30)
chart_cb.pack(pady=10)
tk.Button(tab3, text="Vẽ biểu đồ", command=lambda: ve_bieu_do(chart_cb.get()), bg="#2196F3", fg="white").pack()

# === TAB 4: Phân tích ===
summary_text = tk.Text(tab4, wrap=tk.WORD, width=90, height=25)
summary_text.pack(padx=10, pady=10)

def hien_phan_tich():
    year_avg = df.groupby("Crop_Year")["Production"].mean().round(2)
    top_crop = df.groupby("Crop")["Production"].sum().sort_values(ascending=False).head(3)
    summary = "--- Trung bình sản lượng theo năm ---\n"
    summary += year_avg.to_string()
    summary += "\n\n--- Top 3 cây trồng có sản lượng cao nhất ---\n"
    summary += top_crop.to_string()
    summary_text.delete(1.0, tk.END)
    summary_text.insert(tk.END, summary)

tk.Button(tab4, text="Xem phân tích", command=hien_phan_tich, bg="#795548", fg="white").pack(pady=10)

# === TAB 5: Tra cứu ===
tk.Label(tab5, text="Tìm theo loại cây hoặc năm:").pack(pady=10)
search_entry = ttk.Entry(tab5, width=40)
search_entry.pack()

result_tree = ttk.Treeview(tab5, columns=("Crop", "Year", "Area", "Humidity", "Wind_Speed", "Temperature", "Production"), show='headings')
for col in result_tree["columns"]:
    result_tree.heading(col, text=col)
    result_tree.column(col, width=130)
result_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

def tim_kiem():
    query = search_entry.get().strip().lower()
    result_tree.delete(*result_tree.get_children())
    filtered = df[df["Crop"].str.lower().str.contains(query) | df["Crop_Year"].astype(str).str.contains(query)]
    for _, row in filtered.iterrows():
        result_tree.insert("", tk.END, values=(row["Crop"], row["Crop_Year"], row["Area"],
                                               row["Humidity"], row["Wind_Speed"], row["Temperature"], row["Production"]))

tk.Button(tab5, text="Tìm", command=tim_kiem, bg="#3F51B5", fg="white").pack(pady=10)

# === Chạy ứng dụng ===
root.mainloop()