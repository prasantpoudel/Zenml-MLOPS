import json

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import pickle


def main():
    st.title("End to End Customer Satisfaction Pipeline with ZenML")
    model=  pickle.load(open('model.pkl', 'rb'))
    st.markdown(
        """ 
    #### Problem Statement 
     The objective here is to predict the customer satisfaction score for a given order based on features like order status, price, payment, etc. I will be using [ZenML](https://zenml.io/) to build a production-ready pipeline to predict the customer satisfaction score for the next order or purchase.    """
    )

    st.markdown(
        """ 
    Above is a figure of the whole pipeline, we first ingest the data, clean it, train the model, and evaluate the model, and if data source changes or any hyperparameter values changes, deployment will be triggered, and (re) trains the model and if the model meets minimum accuracy requirement, the model will be deployed.
    """
    )

    st.markdown(
        """ 
    #### Description of Features 
    This app is designed to predict the customer satisfaction score for a given customer. You can input the features of the product listed below and get the customer satisfaction score. 
    | Models        | Description   | 
    | ------------- | -     | 
    | Payment Sequential | Customer may pay an order with more than one payment method. If he does so, a sequence will be created to accommodate all payments. | 
    | Payment Installments   | Number of installments chosen by the customer. |  
    | Payment Value |       Total amount paid by the customer. | 
    | Price |       Price of the product. |
    | Freight Value |    Freight value of the product.  | 
    | Product Name length |    Length of the product name. |
    | Product Description length |    Length of the product description. |
    | Product photos Quantity |    Number of product published photos |
    | Product weight measured in grams |    Weight of the product measured in grams. | 
    | Product length (CMs) |    Length of the product measured in centimeters. |
    | Product height (CMs) |    Height of the product measured in centimeters. |
    | Product width (CMs) |    Width of the product measured in centimeters. |
    """
    )
    payment_sequential = st.sidebar.slider("Payment Sequential")
    payment_installments = st.sidebar.slider("Payment Installments")
    payment_value = st.number_input("Payment Value",value=0)
    price = st.number_input("Price",value=0)
    freight_value = st.number_input("freight_value",value=0)
    product_name_length = st.number_input("Product name length",    value=0)
    product_description_length = st.number_input("Product Description length",value=0)
    product_photos_qty = st.number_input("Product photos Quantity ",    value=0)
    product_weight_g = st.number_input("Product weight measured in grams",  value=0)
    product_length_cm = st.number_input("Product length (CMs)",value=0)
    product_height_cm = st.number_input("Product height (CMs)",value=0)
    product_width_cm = st.number_input("Product width (CMs)",value=0)

    if st.button("Predict"):
        
        features = [[payment_sequential,payment_installments,payment_value,price,freight_value,product_name_length,product_description_length,product_photos_qty,product_weight_g,product_length_cm,product_height_cm,product_width_cm]]
        predict=model.predict(features)
        st.success(
            "Your Customer Satisfactory rate(range between 0 - 5) with given product details is :-{}".format(
                predict
            )
        )
if __name__ == "__main__":
    main()