import os
import cv2
import numpy as np

import streamlit as st
import pandas as pd
#detection : import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import xlsxwriter
import argparse
import imutils

def create_excel(rows, folder_name):
    columns = ["Folder","Image Name", "Average area of granules", "Size of largest area", "Size of smallest area", "Average perimeter of granules", "Size of largest perimeter", "Size of smallest perimeter", "Average Red of Granules", "Average Green of Granules", "Average Blue of Granules"]
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
        for j in range (11):
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

def setup_folder_categories():
    folder_names = dict()
    folder_names["5,7 rpm"] = 1
    folder_names["5,7 rpm vissza"] = 2
    folder_names["6,1 rpm"] = 3
    folder_names["6,1 rpm vissza"] = 4
    folder_names["6,5 rpm"] = 5
    return folder_names

def prepare_dataset(folder_names):
    folder_dir = os.getcwd() + '/RotationCNN' + '/Training'

    for folder_name in os.listdir(folder_dir):
        st.write('aaaaaa')
        rows = []
        images_dir = folder_dir + '/' + folder_name
        for image in os.listdir(images_dir):
            if (image.endswith(".jpg")):
                img = cv2.imread(images_dir + "/" + image)
                #cv2.imshow('Original', img)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                thr, gray1 = cv2.threshold(gray, 80 ,140, cv2.THRESH_BINARY)
                contours, hierarchy = cv2.findContours(gray1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                print("Number of Contours found = " + str(len(contours)))
                #cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

                #cv2.imshow('Contours', img)
                #cv2.waitKey(0)
                
                final = np.zeros(img.shape,np.uint8)
                mask = np.zeros(gray.shape,np.uint8)
                
                res_areas = []
                res_perims = []
                res_min_areas = []
                colors = []
                i = 0
                rgb_line1 = []
                rgb_line2 = []
                rgb_line3 = []
                
                for cntr in contours:
                    mask[...]=0
                    
                    #cv2.drawContours(mask,contours,i,255,-1)
                    #cv2.drawContours(final,contours,i,cv2.mean(img,mask),-1)
                    #print(cv2.mean(img, mask))
                    
                    temp = cv2.mean(img, mask)
                    rgb_line1.append(temp[0])
                    rgb_line2.append(temp[1])
                    rgb_line3.append(temp[2])
                
                    i = i+1
                    
                    perimeter = cv2.arcLength(cntr, True)
                    res_perims.append(perimeter)
                    
                    rect = cv2.minAreaRect(cntr)
                    res_min_areas.append(rect[1][0] * rect[1][1])
                    
                    x, y, w, h = cv2.boundingRect(cntr)
                    res_areas.append(w*h)

                res_areas.sort()
                res_min_areas.sort()
                res_perims.sort()
                #print(res_areas)
                #print(res_min_areas)

                
                #cv2.imshow('im',img)
                #cv2.imshow('final',final)
                
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

                # Features Calculations
                # todo
                if sum(res_perims) > 0 and len(res_perims) > 0 and sum(res_min_areas) > 0 and len(res_min_areas) > 0 and len(rgb_line1) > 0 and len(rgb_line2) > 0 and len(rgb_line3) > 0 :
                    largest_area = res_min_areas[-1]
                    smallest_area = res_min_areas[0]
                    largest_perim = res_perims[-1]
                    smallest_perim = res_perims[0]
                    average_area = sum(res_min_areas) / len(res_min_areas)
                    average_perimeter = sum(res_perims) / len(res_perims)
                    rgb_avg1= sum(rgb_line1) / len(rgb_line1)
                    rgb_avg2= sum(rgb_line2) / len(rgb_line2)
                    rgb_avg3= sum(rgb_line3) / len(rgb_line3)
                    #print(rvg_avg)
                    rows.append([folder_names[folder_name],image, average_area, largest_area, smallest_area, average_perimeter, largest_perim, smallest_perim, rgb_avg1, rgb_avg2, rgb_avg3])

                
        
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