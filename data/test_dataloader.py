from data_loader import ModelNet40
import os


dataset_path = os.path.join(os.getcwd(), 'ModelNet40')

train_loader = ModelNet40(dataset_path=dataset_path, test=False)


data, label, label_txt = train_loader[0]

print(f"Datapoint label: {label} | txt: {label_txt}")



# NOTE: The below code is meant to test if all the images in the ModelNet40
# are working fine.


# errors = 0
# corrupted_data = []
# corrupted_data_indecies = []


# for i in range(len(train_loader)):
# # for i in range(50):
#     try:
#         train_loader[i]
#     except Exception as e:
#         errors += 1
#         print(f'WARNING: {train_loader.data_points_paths[i][0]} has errored out')
#         corrupted_data.append(train_loader.data_points_paths[i][0])
#         corrupted_data_indecies.append(i)


# print(f'Number of corrupted datapoints {errors}')

# file_name = "corrupted_images.txt"
# file_path = os.path.join(os.getcwd(), file_name)
# with open(file_path, 'w') as file:
#     for point in corrupted_data:
#         file.write(point)
#         file.write('\n')
        
# file_name = "corrupted_images_indecies.txt"
# file_path = os.path.join(os.getcwd(), file_name)
# with open(file_path, 'w') as file:
#     for point in corrupted_data_indecies:
#         file.write(str(point))
#         file.write('\n')