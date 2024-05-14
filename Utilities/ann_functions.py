import os
import cv2
import numpy as np

from PIL import Image
import streamlit as st
import pandas as pd

import xlsxwriter
from sklearn.model_selection import train_test_split

def create_excel(rows, folder_name):
    columns = ["Folder","Image Name", "Average area of granules", "Size of largest area", "Size of smallest area", "Average perimeter of granules", "Size of largest perimeter", "Size of smallest perimeter", "Average Blue of Granules", "Average Green of Granules", "Average Red of Granules", "Average Solidity", "Average Orientation Angle"]
    # Create a workbook and add a worksheet.
    workbook = xlsxwriter.Workbook(folder_name + '.xlsx')
    worksheet = workbook.add_worksheet()

    # Start from the first cell. Rows and columns are zero indexed.
    row = 0
    col = 0
    for i in range(len(columns)):
        worksheet.write(row,col + i, columns[i])
    
    row = row + 1 
    # Iterate over the data and write it out row by row.
    for i in range(len(rows)):
        for j in range (len(columns)):
            worksheet.write(row + i, col + j, rows[i][j])
            
    
    workbook.close()

def merge_excels():
    excel_dir = os.getcwd()
    df=pd.DataFrame()
    for dfs in os.listdir(excel_dir):  
        if (dfs.endswith(".xlsx")):
            dfs = pd.read_excel(excel_dir+'/' + dfs )
            dfs.drop('Image Name', inplace = True , axis = 1)
            df=pd.concat([df,dfs])
    df.to_csv('merged_dataset.csv')

def setup_folder_categories(all_categories):
    folder_names = dict()
    label = 1
    for category in all_categories:
        folder_names[category] = label
        label = label + 1
    return folder_names

def prepare_dataset(folder_names, category_images_dict):
    for folder_name, images in category_images_dict.items():
        rows = []
        for image in images:
            if (image.name.endswith(".jpg")):
                # Convert the uploaded file to a PIL Image object
                img = Image.open(image)
                img = np.array(img)
                #img = cv2.imread(pil_image)
                #cv2.imshow('Original', img)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                thr, gray1 = cv2.threshold(gray, 80 ,140, cv2.THRESH_BINARY)
                contours, hierarchy = cv2.findContours(gray1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                
                final = np.zeros(img.shape,np.uint8)
                mask = np.zeros(gray.shape,np.uint8)
                
                res_areas = []
                res_perims = []
                #res_min_areas = []
                solidities = []
                all_orientation_angles = []
                i = 0
                blue_colors = []
                green_colors = []
                red_colors = []
                
                for cntr in contours:
                    # *** Colors Part ***
                    # Calculate bounding box for contour
                    x, y, w, h = cv2.boundingRect(cntr)
                    
                    # Extract ROI (Region of Interest)
                    roi = img[y:y+h, x:x+w]
                    
                    # Calculate average color
                    avg_color_per_row = np.average(roi, axis=0)
                    avg_color = np.average(avg_color_per_row, axis=0)
                    
                    # Convert average color to RGB format
                    avg_color_rgb = tuple(map(int, avg_color[::-1]))  # OpenCV stores colors in BGR format
                    
                    #print("Average RGB color:", avg_color_rgb)

                    blue_colors.append(avg_color_rgb[0])
                    green_colors.append(avg_color_rgb[1])
                    red_colors.append(avg_color_rgb[2])
                
                    i = i+1
                    
                    # *** Perimeter Part ***
                    perimeter = cv2.arcLength(cntr, True)
                    res_perims.append(perimeter)
                    
                    # *** Area Part ***
                    contour_area = cv2.contourArea(cntr)
                    res_areas.append(contour_area)

                    # *** Orientation Angle Part ***
                    # Ensure contour has enough points to fit an ellipse
                    if len(cntr) >= 5:
                        # Fit an ellipse to the contour
                        ellipse = cv2.fitEllipse(cntr)
                        orientation_angle = ellipse[2]
                        all_orientation_angles.append(orientation_angle)
                        print("Orientation angle:", orientation_angle)

                    # *** Solidity Part ***
                    # Calculate the convex hull of the contour
                    hull = cv2.convexHull(cntr)
                    
                    # Calculate the area of the convex hull
                    hull_area = cv2.contourArea(hull)
                    
                    if hull_area != 0:
                        # Calculate solidity (contour area / convex hull area)
                        solidity = contour_area / hull_area
                        solidities.append(solidity)
                        # Print the solidity ratio
                        print("Solidity:", solidity)

                #res_areas.sort()
                res_areas.sort()
                res_perims.sort()

                # Features Calculations
                # todo
                if sum(all_orientation_angles) > 0 and len(all_orientation_angles) > 0 and sum(solidities) > 0 and len(solidities) > 0 and sum(res_perims) > 0 and len(res_perims) > 0 and sum(res_areas) > 0 and len(res_areas) > 0 and len(blue_colors) > 0 and len(green_colors) > 0 and len(red_colors) > 0 :
                    largest_area = res_areas[-1]
                    smallest_area = res_areas[0]
                    largest_perim = res_perims[-1]
                    smallest_perim = res_perims[0]
                    average_area = sum(res_areas) / len(res_areas)
                    average_perimeter = sum(res_perims) / len(res_perims)
                    average_solidity = sum(solidities) / len(solidities)
                    average_orientation_angle = sum(all_orientation_angles) / len(all_orientation_angles)
                    print(f"bababababa   {average_orientation_angle}")
                    rgb_avg1= sum(blue_colors) / len(blue_colors) # B
                    rgb_avg2= sum(green_colors) / len(green_colors) # G
                    rgb_avg3= sum(red_colors) / len(red_colors) # R

                    rows.append([folder_names[folder_name],image.name, average_area, largest_area, smallest_area, average_perimeter, largest_perim, smallest_perim, rgb_avg1, rgb_avg2, rgb_avg3, average_solidity, average_orientation_angle])

        create_excel(rows, folder_name)
    merge_excels()
    # Excel file part
    # soft max layer le5ra
    # one hot encoding of the target categories
    # keras

def load_df_from_excel():
    excel_dir = os.getcwd()
    df=pd.DataFrame()
    for dfs in os.listdir(excel_dir):  
        if (dfs.endswith(".xlsx")):
            dfs = pd.read_excel(excel_dir+'/' + dfs )
            dfs.drop('Image Name', inplace = True , axis = 1)
            df=pd.concat([df,dfs])
    return df

def prepare_train_and_test_xy(df, train_size):
    X = df.drop(["Folder"], axis=1)
    y = df["Folder"]
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=round(1 - (train_size * 0.01), 2), random_state=42)

    return X_train, X_test, y_train, y_test


"""
def prepare_train_validation_sets():
    train_dataset = train.flow_from_directory(folder_dir, 
                                          target_size= (200,200),
                                          batch_size = 32,
                                          class_mode= 'categorical')

    validation_dataset = validation.flow_from_directory(folder_dir+'/validation/', 
                                            target_size= (200,200),
                                            batch_size = 32,
                                            class_mode= 'categorical')
    
    return train_dataset, validation_dataset

"""