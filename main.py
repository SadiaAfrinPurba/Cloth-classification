
import cv2
from PIL import Image #needs because of displaying the images in GUI window
from PIL import ImageTk
from tkinter import Tk, Label, filedialog, Button
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from skimage import exposure
from skimage import feature
from imutils import paths


def selectImage():
    global panelInputImg,panelOutputImg,panelClass,panelCannyImage
    path = filedialog.askopenfilename()
    
    
    if len(path) > 0:
        inputImage = cv2.imread(path)
        resizeInputImage = cv2.resize(inputImage, (256, 256))
        grayInputImage = cv2.cvtColor(inputImage,cv2.COLOR_BGR2GRAY)
        cannyImage = cv2.Canny(grayInputImage, 50, 100) 
        resizeCannyImage = cv2.resize(cannyImage, (256, 256))
        logo = cv2.resize(grayInputImage, (200, 100))

        # H is the feature vector
        (H, hogImage) = feature.hog(logo, orientations=9, pixels_per_cell=(10, 10),
		cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualise=True)
        hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
        hogImage = hogImage.astype("uint8")
        
        # load training model
        model_load = joblib.load('saved_model.pkl')
        pred = model_load.predict(H.reshape(1, -1))[0]
       
        
        #converts openCv to PIL format
        inputImage = Image.fromarray(resizeInputImage)
        cannyImage = Image.fromarray(resizeCannyImage)
        hogImage = Image.fromarray(hogImage)
        
        #converts PIL to ImageTK format
        inputImage = ImageTk.PhotoImage(inputImage)
        cannyImage = ImageTk.PhotoImage(cannyImage)
        hogImage = ImageTk.PhotoImage(hogImage)
        
        
        if panelInputImg is None or panelOutputImg is None:
            panelInputImg = Label(image = inputImage)
            panelInputImg.image = inputImage #To prevent Python’s garbage collection routines from deleting the image
            panelInputImg.pack(side="left", padx=15, pady=15)

            panelOutputImg = Label(image = hogImage)
            panelOutputImg.image = hogImage
            panelOutputImg.pack(side="right", padx=15, pady=15)

            panelCannyImage = Label(image = cannyImage)
            panelCannyImage.image = cannyImage
            panelCannyImage.pack(side="right", padx=15, pady=15)
            
            panelClass = Label(text = pred.title(),font=("Arial Bold", 50)) 
            panelClass.text = pred.title()
            panelClass.pack(side="top", padx=15, pady=15)


        else:
            panelInputImg.configure(image = inputImage)
            panelCannyImage.configure(image = cannyImage)
            panelOutputImg.configure(image = hogImage)
            panelClass.configure(text = pred.title(),font=("Arial Bold", 50))
            panelInputImg.image = inputImage,
            panelCannyImage.image = cannyImage
            panelOutputImg.image = hogImage
            panelClass.text = pred.title()
            
#Initializing the window
root = Tk()
root.title('Cloth Classification')
# root.geometry('200x100')
panelInputImg = None
panelOutputImg = None
panelCannyImage = None
panelClass = None
btn = Button(root, text="Select an image", command=selectImage)
btn.pack(side="bottom", fill="both", expand="no", padx="10", pady="10")

root.mainloop()
            
            
            
        
             



