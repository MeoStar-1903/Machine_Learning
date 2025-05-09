new_house = np.array([[100, 1, 3, 1], [150, 0, 4, 1], [80, 1, 2, 0]])

# Thêm cột bias vào new_data
new_house_with_bias = np.hstack([np.ones((new_house.shape[0], 1)), new_house]) 