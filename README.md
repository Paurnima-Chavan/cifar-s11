# CIFAR-10 Dataset PyTorch implementation using Resnet18 and Grad-CAM

In this project, we trained a ResNet18 model on the CIFAR10 dataset for 20 epochs. To gain deeper insights into the misclassifications, we used Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize the regions of the images that were most influential in the model's decision-making process.

## Code organization

Code organization in this project is structured into five files.

src

&nbsp;&nbsp;&nbsp;&nbsp;models

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `models.py`


&nbsp;&nbsp;&nbsp;&nbsp; `dataset.py`
 
&nbsp;&nbsp;&nbsp;&nbsp; `pytorch_grad_cam.py`

&nbsp;&nbsp;&nbsp;&nbsp; `test.py`

&nbsp;&nbsp;&nbsp;&nbsp; `train.py`
 
&nbsp;&nbsp;&nbsp;&nbsp; `utils.py`

`main.py`

`S11.ipynb`

The file **"main.py"** contains code for loading the cifar10 dataset, loading and training model.
**"S11.ipynb"** acts as the notebook where the actual execution and experimentation take place, and it just calls single function 

## Model Overview

- **Model Summary**:
  
     ![image](https://github.com/Paurnima-Chavan/cifar-s11/assets/25608455/331c8bd1-d349-47d8-8b49-5012ece66cf3)

- **Model Training Summary**:

   ![image](https://github.com/Paurnima-Chavan/cifar-s11/assets/25608455/8c01e6f6-1493-437b-8af6-ac94480fd096)


- **Model loss curves for test and train datasets**:
    ![image](https://github.com/Paurnima-Chavan/cifar-s11/assets/25608455/6420e42f-54c8-4825-a998-a5c5cba8876d)

  
## **Transformations**: 

  ![image](https://github.com/Paurnima-Chavan/cifar-s11/assets/25608455/9fe4b797-25e3-4445-9817-840954a0c50b)


## 20 misclassified images

  ![image](https://github.com/Paurnima-Chavan/cifar-s11/assets/25608455/e5f45e1a-03c3-4572-a3ef-ce8b2414fddc)

## 20 GradCam outputs on any misclassified images

  ![image](https://github.com/Paurnima-Chavan/cifar-s11/assets/25608455/66bf999a-7279-4021-a76a-31cbe0ff938f)


## Conclusion
To gain deeper insights into the misclassifications, we used Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize the regions of the images that were most influential in the model's decision-making process. This technique helps to highlight the areas that the model focused on when making its predictions, providing valuable information for model interpretation and debugging.

By combining these analyses, we gained a comprehensive understanding of the model's performance, its strengths, and potential areas for improvement. The results and visualizations obtained from this project are crucial for further optimizing the model and enhancing its accuracy for image classification tasks.
