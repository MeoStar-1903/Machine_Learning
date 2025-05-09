import pandas as pd
import numpy as np

#Đọc dữ liệu từ file csv
df = pd.read_csv("Data/gia_nha.csv")

#Chuẩn bị dữ liệu
X = df[["Dien_tich_m2", "Co_giay_to", "So_phong", "Co_noi_that"]].values
y = df["Gia_nha_ty"].values.reshape(-1, 1)

X_bias = np.hstack([np.ones((X.shape[0], 1)), X])  # Thêm cột bias vào X

#Tính hệ số beta the công thức đóng
XtX = X_bias.T.dot(X_bias)
XtX_inv = np.linalg.inv(XtX)
Xty = X_bias.T.dot(y)
beta = XtX_inv.dot(Xty)

#Dự đoán giá nhà
y_pred = X_bias.dot(beta)

#Đánh giá mô hình
mse = np.mean((y - y_pred) ** 2)
ss_total = np.sum((y - np.mean(y)) ** 2)
ss_res = np.sum((y - y_pred) ** 2)
r2 = 1 - (ss_res / ss_total)

#Print đánh giá mô hình
#print("Hệ số beta (bao gồm intercept):")
#print(beta.flatten())
#print("Mean squared Error:", mse)
#print ("R^2 score:", r2)

#Dự đoán giá nhà cho một số mẫu mới
new_house = np.array([[100, 1, 3, 1], [150, 0, 4, 1], [80, 1, 2, 0]])

# Thêm cột bias vào new_data
new_house_with_bias = np.hstack([np.ones((new_house.shape[0], 1)), new_house]) 

predict = new_house_with_bias.dot(beta)
print("Dự đoán giá nhà cho các mẫu mới(tỷ): ")
print(predict)