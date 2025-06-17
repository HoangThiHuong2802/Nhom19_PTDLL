import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt


# Hàm đọc và làm sạch dữ liệu
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna()

    # Đảm bảo kiểu dữ liệu số
    num_cols = ['Production', 'Area', 'Temperature', 'Humidity', 'Wind_Speed', 'Crop_Year']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df[df[col].notna() & (df[col] > 0)]

    return df

def main():
    file_path = 'Crop_production_in_India_ok.csv'
    df = load_and_clean_data(file_path)

    # Mã hóa one-hot các cột phân loại
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Tách dữ liệu đầu vào và đầu ra
    X = df_encoded.drop(columns=['Production'])
    y = df_encoded['Production']

    # Lưu danh sách cột đã mã hóa để dự đoán
    encoded_columns = X.columns

    # Chia dữ liệu train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Huấn luyện mô hình
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Đánh giá mô hình
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    print("\n=== Kết quả mô hình ===")
    print(f"Train R-squared: {train_r2:.4f}")
    print(f"Test R-squared:  {test_r2:.4f}")
    print(f"Train MSE:       {train_mse:.2f}")
    print(f"Test MSE:        {test_mse:.2f}")

    # =====================
    # Dự đoán với dữ liệu người dùng
    # =====================
    try:
        print("\n=== Nhập dữ liệu để dự đoán sản lượng ===")
        user_input = {
            'Crop_Year': int(input("Năm trồng (Crop_Year): ")),
            'Area': float(input("Diện tích (Area): ")),
            'Temperature': float(input("Nhiệt độ (°C): ")),
            'Humidity': float(input("Độ ẩm (%): ")),
            'Wind_Speed': float(input("Tốc độ gió (km/h): ")),
            'Crop': input("Loại cây trồng (Crop): ").strip().title(),
            'Season': input("Mùa vụ (Season): ").strip().title()
        }

        user_df = pd.DataFrame([user_input])
        user_df_encoded = pd.get_dummies(user_df)

        # Thêm các cột còn thiếu
        for col in encoded_columns:
            if col not in user_df_encoded.columns:
                user_df_encoded[col] = 0

        # Sắp xếp đúng thứ tự cột
        user_df_encoded = user_df_encoded[encoded_columns]

        # Dự đoán
        prediction = model.predict(user_df_encoded)[0]
        print(f"\n🔮 Dự đoán sản lượng: {prediction:.2f} tấn")

    except Exception as e:
        print(f"\n❌ Lỗi khi nhập dữ liệu: {e}")

    # =====================
    # Biểu đồ
    # =====================
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train['Area'], y_train, alpha=0.4, label='Train')
    plt.scatter(X_test['Area'], y_test, alpha=0.4, label='Test')
    plt.xlabel("Diện tích (ha)")
    plt.ylabel("Sản lượng (tấn)")
    plt.title("Biểu đồ: Diện tích vs Sản lượng")
    plt.legend()
    plt.savefig("plot_area_vs_production.png")
    print("✅ Biểu đồ đã lưu: plot_area_vs_production.png")

    # Biểu đồ sản lượng theo năm
    plt.figure(figsize=(10, 6))
    plt.plot(df['Crop_Year'], df['Production'], 'o', alpha=0.3)
    plt.xlabel("Năm trồng")
    plt.ylabel("Sản lượng")
    plt.title("Biểu đồ: Crop_Year vs Production")
    plt.savefig("plot_crop_year_vs_production.png")
    print("✅ Biểu đồ đã lưu: plot_crop_year_vs_production.png")

# =======================
# Gọi hàm main
# =======================
if __name__ == "__main__":
    main()
