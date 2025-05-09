import pandas as pd
import numpy as np

#Đọc dữ liệu từ file csv
df = pd.read_csv("Data/gia_nha_dat_re.csv")

#Chuẩn bị dữ liệu
X = df[["Dien_tich_m2", "Co_giay_to", "So_phong", "Co_noi_that"]].values
y = df["Dat_hay_re"].values.reshape(-1, 1)

#Normalize dữ liệu
X = (X - X.mean(axis=0)) / X.std(axis=0)

#Thêm cột bias vào X
X_bias = np.hstack([np.ones((X.shape[0], 1)), X])  # Thêm cột bias vào X

#Khởi tạo trọng số
theta = np.zeros((X_bias.shape[1], 1))

#Cac hàm sigmoid và cost function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    epsilon = 1e-5  # Tránh chia cho 0
    return (-1 / m) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))

#Hàm gradient descent
def gradient_descent(X, y, theta, lr=0.1, epochs=1000):
    m = len(y)
    for i in range(epochs):
        h = sigmoid(X_bias @ theta)
        gradient = (1 / m) * (X.T @ (h - y))
        theta -= lr * gradient
        if i % 100 == 0:
            loss = cost_function(X, y, theta)
            #print(f"Epoch {i}, Loss: {loss:4f}")
    return theta

#Chạy gradient descent
theta = gradient_descent(X_bias, y, theta, lr=0.1, epochs=1000)

#Dự đoán giá nhà
def predict(X, theta, threshold=0.5):
    prob = sigmoid(X @ theta)
    return (prob >= threshold).astype(int)

#Danh giá mô hình
y_pred = predict(X_bias, theta)
accuracy = np.mean(y_pred == y) * 100
print(f"Accuracy: {accuracy:.2f}%")

def du_doan_can_ho(dien_tich, co_giay_to, so_phong, co_noi_that, X_mean, X_std, theta):
    """
    Dự đoán khả năng mua căn hộ dựa trên mô hình Logistic Regression đã huấn luyện.

    Tham số:
    - dien_tich: diện tích căn hộ (m2)
    - co_giay_to: 1 nếu có giấy tờ, 0 nếu không
    - so_phong: số phòng ngủ
    - co_noi_that: 1 nếu có nội thất, 0 nếu không
    - X_mean: trung bình của các đặc trưng từ tập huấn luyện
    - X_std: độ lệch chuẩn của các đặc trưng
    - theta: vector trọng số đã huấn luyện

    Trả về:
    - "Đắt" hoặc "Rẻ" tùy thuộc vào dự đoán của mô hình.
    """
    x_input = np.array([[dien_tich, co_giay_to, so_phong, co_noi_that]])
    x_scaled = (x_input - X_mean) / X_std
    x_bias = np.hstack([np.ones((1, 1)), x_scaled])
    y_pred = predict(x_bias, theta)
    return "Đắt" if y_pred[0][0] == 1 else "Rẻ"

# Sau khi huấn luyện xong và đã tính X_mean, X_std:
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)

# Dự đoán cho căn hộ mới: 75m², có giấy tờ, 3 phòng, không nội thất
ket_qua = du_doan_can_ho(5, 0, 1, 0, X_mean, X_std, theta)
print("Dự đoán:", ket_qua)
