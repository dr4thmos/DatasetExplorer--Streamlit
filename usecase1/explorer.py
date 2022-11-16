import os
import streamlit as st
import numpy as np
import cv2 as cv

from astropy.io import fits

from database import DataBase, RadioDB

import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

radaba = RadioDB() # RAdioDAtaBAse -> radaba

radaba.data_folder = "data"
radaba.compact_folder = "compact"
radaba.extended_folder = "extended"
radaba.multi_island_folder = "extended-multisland"

compact_path = os.path.join(radaba.data_folder, radaba.compact_folder)
extended_path = os.path.join(radaba.data_folder, radaba.extended_folder)
multi_island_path = os.path.join(radaba.data_folder, radaba.multi_island_folder)

check_histogram = st.button("Histogram visualization")

for folder in os.listdir(compact_path):
    sample_folder = os.path.join(compact_path, folder, "masked_imgs")
    for source_fit in os.listdir(sample_folder):
        st.write(source_fit)
        raw_column, minmax_column, global_equalized_column, equalizer_column  = st.columns(4)

        file_path = os.path.join(sample_folder, source_fit)
        hdul = fits.open(file_path)
        npy_image = hdul[0].data

        
        
        # Original Histogram
        with raw_column:
            st.title("RAW")
            if check_histogram:
                _ = plt.hist(npy_image, bins='auto')  # arguments are passed to np.histogram
                plt.title("Raw Histogram")
                st.pyplot(plt)
        
        # MinMax Histogram
        with minmax_column:
            st.title("MINMAX")
            lower_bound = np.amin(npy_image)
            upper_bound = np.amax(npy_image)
            npy_image_minmax = (npy_image - lower_bound) / (upper_bound - lower_bound)
            
            if check_histogram:
                _ = plt.hist(npy_image_minmax, bins='auto')  # arguments are passed to np.histogram
                plt.title("Min Max Histogram")
                st.pyplot(plt)

            st.image(npy_image_minmax)

            if check_histogram:
                st.write("Min: {}".format(lower_bound))
                st.write("Max: {}".format(upper_bound))
            
        # Equalizer Histogram CLAHE (Contrast Limited Adaptive Histogram Equalization) 
        with equalizer_column:
            st.title("CLAHE")
            clahe = cv.createCLAHE()
            cl1 = clahe.apply(np.uint8(npy_image_minmax*255))

            if check_histogram:
                _ = plt.hist(cl1, bins='auto')  # arguments are passed to np.histogram
                plt.title("CLAHE Equalized  Histogram")
                st.pyplot(plt)
            
            st.image(cl1)


        # Global equalized Histogram
        with global_equalized_column:
            st.title("GLOBEQ")
            equ = cv.equalizeHist(np.uint8(npy_image_minmax*255))
            
            if check_histogram:    
                _ = plt.hist(equ, bins='auto')  # arguments are passed to np.histogram
                plt.title("Global Equalized  Histogram")
                st.pyplot(plt)
            
            st.image(equ)

        break

for folder in os.listdir(extended_path):
    sample_folder = os.path.join(extended_path, folder, "masked_imgs")
    for source_fit in os.listdir(sample_folder):
        st.write(source_fit)
        raw_column, minmax_column, global_equalized_column, equalizer_column  = st.columns(4)

        file_path = os.path.join(sample_folder, source_fit)
        hdul = fits.open(file_path)
        npy_image = hdul[0].data

        
        
        # Original Histogram
        with raw_column:
            st.title("RAW")
            if check_histogram:
                _ = plt.hist(npy_image, bins='auto')  # arguments are passed to np.histogram
                plt.title("Raw Histogram")
                st.pyplot(plt)
        
        # MinMax Histogram
        with minmax_column:
            st.title("MINMAX")
            lower_bound = np.amin(npy_image)
            upper_bound = np.amax(npy_image)
            npy_image_minmax = (npy_image - lower_bound) / (upper_bound - lower_bound)
            
            if check_histogram:
                _ = plt.hist(npy_image_minmax, bins='auto')  # arguments are passed to np.histogram
                plt.title("Min Max Histogram")
                st.pyplot(plt)

            st.image(npy_image_minmax)

            if check_histogram:
                st.write("Min: {}".format(lower_bound))
                st.write("Max: {}".format(upper_bound))
            
        # Equalizer Histogram CLAHE (Contrast Limited Adaptive Histogram Equalization) 
        with equalizer_column:
            st.title("CLAHE")
            clahe = cv.createCLAHE()
            cl1 = clahe.apply(np.uint8(npy_image_minmax*255))

            if check_histogram:
                _ = plt.hist(cl1, bins='auto')  # arguments are passed to np.histogram
                plt.title("CLAHE Equalized  Histogram")
                st.pyplot(plt)
            
            st.image(cl1)


        # Global equalized Histogram
        with global_equalized_column:
            st.title("GLOBEQ")
            equ = cv.equalizeHist(np.uint8(npy_image_minmax*255))
            
            if check_histogram:    
                _ = plt.hist(equ, bins='auto')  # arguments are passed to np.histogram
                plt.title("Global Equalized  Histogram")
                st.pyplot(plt)
            
            st.image(equ)

        break


for folder in os.listdir(multi_island_path):
    sample_folder = os.path.join(multi_island_path, folder, "masked_imgs")
    for source_fit in os.listdir(sample_folder):
        st.write(source_fit)
        raw_column, minmax_column, global_equalized_column, equalizer_column  = st.columns(4)

        file_path = os.path.join(sample_folder, source_fit)
        hdul = fits.open(file_path)
        npy_image = hdul[0].data

        
        
        # Original Histogram
        with raw_column:
            st.title("RAW")
            if check_histogram:
                _ = plt.hist(npy_image, bins='auto')  # arguments are passed to np.histogram
                plt.title("Raw Histogram")
                st.pyplot(plt)
        
        # MinMax Histogram
        with minmax_column:
            st.title("MINMAX")
            lower_bound = np.amin(npy_image)
            upper_bound = np.amax(npy_image)
            npy_image_minmax = (npy_image - lower_bound) / (upper_bound - lower_bound)
            
            if check_histogram:
                _ = plt.hist(npy_image_minmax, bins='auto')  # arguments are passed to np.histogram
                plt.title("Min Max Histogram")
                st.pyplot(plt)

            st.image(npy_image_minmax)

            if check_histogram:
                st.write("Min: {}".format(lower_bound))
                st.write("Max: {}".format(upper_bound))
            
        # Equalizer Histogram CLAHE (Contrast Limited Adaptive Histogram Equalization) 
        with equalizer_column:
            st.title("CLAHE")
            clahe = cv.createCLAHE()
            cl1 = clahe.apply(np.uint8(npy_image_minmax*255))

            if check_histogram:
                _ = plt.hist(cl1, bins='auto')  # arguments are passed to np.histogram
                plt.title("CLAHE Equalized  Histogram")
                st.pyplot(plt)
            
            st.image(cl1)


        # Global equalized Histogram
        with global_equalized_column:
            st.title("GLOBEQ")
            equ = cv.equalizeHist(np.uint8(npy_image_minmax*255))
            
            if check_histogram:    
                _ = plt.hist(equ, bins='auto')  # arguments are passed to np.histogram
                plt.title("Global Equalized  Histogram")
                st.pyplot(plt)
            
            st.image(equ)

        break