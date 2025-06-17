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
    df['Production'] = pd.to_numeric(df['Production'], errors='coerce')
    df['Area'] = pd.to_numeric(df['Area'], errors='coerce')
    df = df[df['Production'] > 0]
    df = df[df['Area'] > 0]
    return df


# Hàm chính
def main():
    # Đường dẫn đến file CSV (giả sử trong cùng thư mục)
    file_path = 'Crop_production_in_India_ok.csv'

    # Tải và làm sạch dữ liệu
    df = load_and_clean_data(file_path)

    # Chuẩn bị dữ liệu cho hồi quy
    X = df[['Area']].values  # Biến độc lập (diện tích)
    y = df['Production'].values  # Biến phụ thuộc (sản lượng)

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Tạo và huấn luyện mô hình hồi quy tuyến tính
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Dự đoán trên tập huấn luyện và tập kiểm tra
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Tính toán chỉ số đánh giá
    train_r2 = r2_score(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    # In kết quả phân tích
    print("=== Kết Quả Phân Tích Hồi Quy ===")
    print(f"Phương trình hồi quy: Production = {model.coef_[0]:.4f} * Area + {model.intercept_:.4f}")
    print(f"Train R-squared: {train_r2:.4f} (Hiệu suất trên tập huấn luyện)")
    print(f"Train MSE: {train_mse:.4f}")
    print(f"Test R-squared: {test_r2:.4f} (Hiệu suất trên tập kiểm tra)")
    print(f"Test MSE: {test_mse:.4f}")

    # Nhập dữ liệu từ người dùng
    try:
        user_area = float(input("Nhập diện tích (hectares) để dự đoán sản lượng: "))
        if user_area < 0:
            print("Diện tích phải là số dương. Vui lòng nhập lại.")
        else:
            # Dự đoán sản lượng
            predicted_production = model.coef_[0] * user_area + model.intercept_
            print(f"Dự đoán sản lượng cho {user_area} hectares: {predicted_production:.2f} tonnes")
    except ValueError:
        print("Vui lòng nhập một số hợp lệ.")

    # Vẽ biểu đồ
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, color='blue', alpha=0.5, label='Train data')
    plt.scatter(X_test, y_test, color='green', alpha=0.5, label='Test data')
    plt.plot(X_train, y_train_pred, color='red',
             label=f'Regression Line (y = {model.coef_[0]:.4f}x + {model.intercept_:.4f})')
    plt.xlabel('Area (hectares)')
    plt.ylabel('Production (tonnes)')
    plt.title('Linear Regression: Area vs Production (Train/Test Split)')
    plt.legend()
    plt.savefig('regression_user_input_plot.png')
    plt.close()

    print("Biểu đồ đã được lưu: regression_user_input_plot.png")


if __name__ == "__main__":
    main()