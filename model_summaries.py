# -*- coding: utf-8 -*-
"""Model Summaries.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17UbkRkSpUGI7S_A9kq_EwNH1HfgN6bm9

# Model Summaries

### ROC Curve

![alt text](https://i.ibb.co/4FCd4w9/download-5.png)

![alt text](https://i.ibb.co/NFvMTLB/download-6.png)

![alt text](https://i.ibb.co/VWH4Qz0/download-27.png://)

![alt text](https://i.ibb.co/jVrcXDh/download-29.png://)

### ROC Statistics

**Logistic Regression**

Average ROC: .77

ROC at 10% FPR: . 37

**Random Forest**

Average ROC: .76

ROC at 10% FPR: . 38

**Decision Tree**

Average ROC: .70

ROC at 10% FPR: . 32

**GBC**

Average ROC: .72

ROC at 10% FPR: . 37

### PRC

![alt text](https://i.ibb.co/vP68Hs7/download-3.png)

![alt text](https://i.ibb.co/q5nTTPd/download-23.png)

![alt text](https://i.ibb.co/WHq6HHK/download-28.png)

![alt text](https://i.ibb.co/Gnzbs1Q/download-30.png://)

### MAE

**Logistic Regression** Mean Absolute Error: 0.23

**Random Forest** Mean Absolute Error: 0.21

**Decision Tree Mean** Mean Absolute Error: 0.21

**GBC** Mean Absolute Error: .26

### RMSE

**Logistic Regression** Root Mean Sqaured Error: 0.01

**Random Forest** Root Mean Sqaured Error: 0.01

**Decision Tree** Root Mean Sqaured Error: 0.01

**GBC** Root Mean Sqaured Error: 0.01

### Classification Report

**Logistic Regression**

                       precision    recall  f1-score   support

           No Click       0.86      0.99      0.92     25495
           Click          0.59      0.10      0.16      4505

**Random Forest**                 
                 
                       precision    recall  f1-score   support

           No Click       0.86      0.98      0.92     25495
           Click          0.53      0.13      0.21      4505

**Decision Tree**     
     
                     precision    recall  f1-score   support

          No Click       0.87      0.97      0.91     25495
          Click          0.45      0.15      0.23      4505

**GBC**

                       precision    recall  f1-score   support

           No Click       0.83      0.99      0.90      8202
           Click          0.59      0.10      0.17      1798

### Confusion Matrix

**Logistic Regression**

![alt text](https://i.ibb.co/DDyKbQC/download-4.png)

**Random Forest**

![alt text](https://i.ibb.co/bRhkGXT/download-9.png)

**Decision Tree**

![alt text](https://i.ibb.co/xhg0kn6/download-26.png)

**GBC**

![alt text](https://i.ibb.co/gMnKBM6/download-31.png)
"""