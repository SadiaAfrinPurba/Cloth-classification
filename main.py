#!/usr/bin/env python
# coding: utf-8
#ref: https://www.pyimagesearch.com/2016/05/23/opencv-with-tkinter/


import cv2
from PIL import Image #needs because of displaying the images in GUI window
from PIL import ImageTk
from tkinter import Tk, Label, filedialog, Button


def selectImage():
    global panelInputImg,panelOutputImg,panelClass
    path = filedialog.askopenfilename()
    
    if len(path) > 0:
        inputImage = cv2.imread(path)
        grayInputImage = cv2.cvtColor(inputImage,cv2.COLOR_BGR2GRAY)
        outputImage = cv2.Canny(grayInputImage, 50, 100) #will change later
        
        # OpenCV represents images in BGR order; however PIL represents
        # images in RGB order, so we need to swap the channels
        inputImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)
        
        #converts openCv to PIL format
        inputImage = Image.fromarray(inputImage)
        outputImage = Image.fromarray(outputImage)
        
        #converts PIL to ImageTK format
        inputImage = ImageTk.PhotoImage(inputImage)
        outputImage = ImageTk.PhotoImage(outputImage)
        
        if panelInputImg is None or panelOutputImg is None:
            panelInputImg = Label(image = inputImage)
            panelInputImg.image = inputImage #To prevent Python’s garbage collection routines from deleting the image
            panelInputImg.pack(side="left", padx=15, pady=15)
            
            panelOutputImg = Label(image = outputImage)
            panelOutputImg.image = outputImage
            panelOutputImg.pack(side="right", padx=15, pady=15)

            panelClass = Label(text = "Shirt")
            panelClass.text = "Shirt"
            panelClass.pack(side="top", padx=15, pady=15)


        else:
            panelInputImg.configure(image = inputImage)
            panelOutputImg.configure(image = outputImage)
            panelClass.configure(text = "Shirt")
            panelInputImg.image = inputImage
            panelOutputImg.image = outputImage
            panelClass.text = "Shirt"
            
#Initializing the window
root = Tk()
root.title('Cloth Classification')
panelInputImg = None
panelOutputImg = None
panelClass = None
btn = Button(root, text="Select an image", command=selectImage)
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

root.mainloop()
            
            
            
        
             



