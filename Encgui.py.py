
from tkinter import *
from tkinter.ttk import*
from tkinter import messagebox
from tkinter import ttk
from tkinter import filedialog
import cv2
import os
import numpy as np
import random
from numpy import*
from PIL import ImageTk,Image
from os.path import isfile, join
import glob
#import vlc
import tkinter.messagebox

# Create an instance of tkinter frame


win= Tk()
win.title('Privacy Preservation In Surveillance Video Using Selective Encryption.')
#win.configure(background='black')
style=Style()
style.configure('W.TButton', font=('calibri',20,'bold'),foreground='purple')


# Set the size of the tkinter window
win.geometry("1000x700")


def openfn():
    global filename
    filename = filedialog.askopenfilename(title='open')
   

def extract_frames():
   global framecount, framewidth, frameheight,fps,frame
   cam = cv2.VideoCapture(filename)
   framecount=int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
   frameheight=int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
   framewidth=int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
   fps=cam.get(cv2.CAP_PROP_FPS)
   print(frameheight)
   print(framewidth)
   print(fps)
   print(framecount)
   currentframe = 0
   while(currentframe<framecount):
   #while(currentframe<4):
    
         ret,frame = cam.read()
         name = 'frame' + str(currentframe) + '.bmp'
         #print ('Creating...' + name)
         cv2.imwrite(name, frame)
         currentframe += 1      
   cam.release()
   
   messagebox.showinfo("Msg", "Frame extraction completed")  
   img=cv2.imread('frame0.bmp')
   #mg=cv2.imread('C:/Users/ISE-HOD/Documents/proj 21-22/batch 1/codes/data/frame0.bmp')
   cv2.imshow('Frame', img)
   
   #img=cv2.resize(img,(200,200))
   #winname='FRAME'
   #cv2.namedWindow(winname)
   #cv2.moveWindow(winname,400,100)
   #cv2.imshow(winname,img)
   print('Frame extraction over')
   
   




def detect_faces():
   #print('count=',framecount)
   #face_cascade = cv2.CascadeClassifier('C:/Users/ISE-HOD/AppData/Local/Programs/Python/Python38/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
   face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
  

   
   currentframe=0
   cam=cv2.VideoCapture(filename)
   #      print("CAP_PROP_FRAME_COUNT  : '{}'".format(cam.get(cv2.CAP_PROP_FRAME_COUNT)))
   length=int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
   framecount=length
   a=np.zeros((framecount,5),int)
   #framecount=4
 
   while currentframe<framecount:
            name = 'frame' + str(currentframe) + '.bmp'
            img = cv2.imread(name)
            #print ('opening...' + name)
            # Convert into grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            # Draw rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
                
                #print(y,y+h,x,x+w)

            if (h<=150)or(w<=150):
                x=y=h=w=0
                
            a[currentframe,0]=currentframe
            a[currentframe,1]=y
            a[currentframe,2]=y+h
            a[currentframe,3]=x
            a[currentframe,4]=x+w
            #print(currentframe)
            currentframe += 1
            
       
   np.save('coordinates',a)
   messagebox.showinfo("Msg", "Face detection completed")
   cv2.imshow('FACE',img)
   
   #img=cv2.imread('C:/Users/ISE-HOD/Documents/proj 21-22/batch 1/codes/data/frame0.bmp')
   #img=cv2.resize(img,(200,200))
   #winname='FACE'
   #cv2.namedWindow(winname)
   #cv2.moveWindow(winname,400,100)
   #cv2.imshow(winname,img)
   
   print('Face detection over')

def encrypt_rowvalcolval():
    
    key1=int(t2.get())

    #random.seed(seed)

    b=np.empty((framecount,5))
    b=np.load('coordinates.npy')
    i=0
    length=framecount
   # while i<length:
    
    while i<length:
        framenumbers=b[i,0]
        r1=b[i,1]
        r2=b[i,2]
        c1=b[i,3]
        c2=b[i,4]
        name = 'frame'+ str(framenumbers) + '.bmp'
        name1 ='frameenc'+ str(framenumbers) + '.bmp'

        #name = 'C:/Users/ISE-HOD/Documents/proj 21-22/batch 1/codes/data/frame'+ str(framenumbers) + '.bmp'
        #name1 ='C:/Users/ISE-HOD/Documents/proj 21-22/batch 1/codes/data/dataenc/frameenc'+ str(framenumbers) + '.bmp'
        
        img=cv2.imread(name)
        #cv2.imshow('1',img)

        rows=r2-r1
        cols=c2-c1
        newimg=img

        np.random.seed(key1)
        rowval=np.random.randint(0,255, size=(rows), dtype=np.uint8)
        rowind=np.argsort(rowval)
        newimg[r1:r2,c1:c2]=img[rowind+r1,c1:c2]
        cv2.imshow('Row shuffle',newimg)
        
        finalimg=newimg
        np.random.seed(key1)
        colval=np.random.randint(0,255, size=(cols), dtype=np.uint8)
        colind=np.argsort(colval)
        finalimg[r1:r2,c1:c2]=newimg[r1:r2,colind+c1]
        cv2.imwrite(name1,finalimg)
        i=i+1
    
    messagebox.showinfo("Msg", "Fist stage encryption completed")
    

    cv2.imshow('Row and Column shuffle',finalimg)
    print('First stage encryption over')


        
    
def encrypt_face():

    #global t3
    key2=float(t3.get())
    
   #value = np.value(0,255, size=(512,512), dtype=np.uint8)
    def value1():
        
        sz=1000*1000*3
        x=np.zeros(sz,dtype=float)
        y=np.zeros(sz,dtype=int)
        value=np.zeros(sz,dtype=int)
        #x[0]=0.003
        x[0]=key2
        r=3.14
        for i in range(1,sz):
            x[i]=x[i-1]*r*(1-x[i-1])
            #print(x[i])
            y[i]=int((x[i]*10**5)%255)
        value=np.reshape(y,(1000,1000,3))
        return value

    print('Key matrix generated')
    
    #print(value[1:5,1:5])
    b=np.empty((framecount,5))
    b=np.load('coordinates.npy')
    i=0
    length=framecount
    value=value1()
    while i<length:
    #while i<1:
        framenumbers=b[i,0]
        r1=b[i,1]
        r2=b[i,2]
        c1=b[i,3]
        c2=b[i,4]
        
        name = 'frameenc'+ str(framenumbers) + '.bmp'
        name1 ='frameencc'+ str(framenumbers) + '.bmp'

        #name = 'C:/Users/ISE-HOD/Documents/proj 21-22/batch 1/codes/data/dataenc/frameenc'+ str(framenumbers) + '.bmp'
        #name1 ='C:/Users/ISE-HOD/Documents/proj 21-22/batch 1/codes/data/dataenc/frameencc'+ str(framenumbers) + '.bmp'

        #print(name)
        img = cv2.imread(name)
        #print(img.shape)
        #print(img)
        #print(r1,r2,c1,c2)
        encimg=np.array(img)
               
        encimg[r1:r2,c1:c2,0] = encimg[r1:r2,c1:c2,0]^value[r1:r2,c1:c2,0]
        encimg[r1:r2,c1:c2,1] = encimg[r1:r2,c1:c2,1]^value[r1:r2,c1:c2,1]
        encimg[r1:r2,c1:c2,2] = encimg[r1:r2,c1:c2,2]^value[r1:r2,c1:c2,2]
        
        img[r1:r2,c1:c2]=encimg[r1:r2,c1:c2]
        cv2.imwrite(name1,encimg)
        i=i+1
        
    messagebox.showinfo("Msg", "Encryption completed")
    cv2.imshow('Final Encryped image',encimg)
    

    #img=cv2.imread('C:/Users/Lenovo/dataenc/frameenc0.bmp')
    #img=cv2.resize(img,(200,200))
    #winname='ENCRYPTED FACE'
    #cv2.namedWindow(winname)
    #cv2.moveWindow(winname,400,100)
    #cv2.imshow(winname,img)
    print('Second stage encryption over')


    

def make_video():
    #print(framecount)
    #print(framewidth)
    #print(frameheight)
    #print(fps)
  
    img_array = []
    
    #for filename in glob.glob('C:/Users/ISE-HOD/Documents/proj 21-22/batch 1/codes/data/dataenc/frameencc*.bmp'):
        
    for filename in glob.glob('frameencc*.bmp'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
        


    out = cv2.VideoWriter('Encvideonew.avi',cv2.VideoWriter_fourcc(*'DIVX'), 29, size)
 
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    tkinter.messagebox.showinfo("Msg","Video make over")

    
    print('video make over')

def close():
    win.destroy()
    
    



# Add an optional Label widget
t1=Label(win, text= "Privacy Preservation In Surveillance Video Using Selective Encryption", font= ('Aerial 25 bold italic')).pack(pady= 30)
#Label(win, text= "Enter secret key 1", font= ('calibri',15,'bold'),foreground='blue').pack(pady=265)
Label(win, text= "Enter secret key 1", font= ('calibri',15,'bold'),foreground='blue').place(x=560,y=370)
Label(win, text= "Enter secret key 2", font= ('calibri',15,'bold'),foreground='blue').place(x=560,y=450)

global t2
t2=Entry(win,font="10",show="*")
t2.place(x=790,y=370, width= 50, height= 30)
global t3
t3=Entry(win,font="10",show="*")
t3.place(x=790,y=450, width= 50, height= 30)


   

# Create a Button to display the message
btn1=ttk.Button(win, text= "Select a Video",style='W.TButton',command=openfn)

btn2=ttk.Button(win, text= "Extract Frames",style='W.TButton',command=extract_frames,compound="right")
   
btn3=ttk.Button(win, text= "Detect Face",style='W.TButton',command=detect_faces,compound="right")

btn4=ttk.Button(win, text= "Encrypt(Position Permutation)",style='W.TButton',command=encrypt_rowvalcolval)

btn5=ttk.Button(win, text= "Encrypt(Pixel Scrambling)",style='W.TButton',command=encrypt_face)

btn6=ttk.Button(win, text= "Play Video",style='W.TButton',command=make_video)

btn7=ttk.Button(win, text= "Exit",style='W.TButton',command=close)

btn1.place(x=100,y=180)
btn2.place(x=100,y=250)
btn3.place(x=100,y=310)
btn4.place(x=100,y=370)
btn5.place(x=100,y=450)
btn6.place(x=100,y=520)
btn7.place(x=970,y=600)

win.mainloop()

