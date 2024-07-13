
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
from PIL import Image,ImageTk
from os.path import isfile, join
import glob
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
   global framecount, framewidth, frameheight,fps
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
         ret,frame = cam.read()
         #name = './datadec/frame' + str(currentframe) + '.bmp'
         name = './data/datadec/frame' + str(currentframe) + '.bmp'
         #print ('Creating...' + name)
         cv2.imwrite(name, frame)
         currentframe += 1      
   cam.release()
   img=cv2.imread('frameenc0.bmp')
   #img=cv2.imread('C:/Users/ISE-HOD/Documents/proj 21-22/batch 1/codes/data/datadec/frame0.bmp')
   cv2.imshow('FRAME',img)

   #img=cv2.resize(img,(200,200))
   #winname='FRAME'
   #cv2.namedWindow(winname)
   #cv2.moveWindow(winname,400,100)
   #cv2.imshow(winname,img)
   tkinter.messagebox.showinfo("Msg", "Frame extraction over")
   print('Frame extraction over')




    
def decrypt_face():
    key2=float(t3.get())
    
    #value = np.random.randint(0,255, size=(915,915), dtype=np.uint8)
    def value1():
        
        sz=1000*1000*3
        x=np.zeros(sz,dtype=float)
        y=np.zeros(sz,dtype=int)
        value=np.zeros(sz,dtype=int)
        x[0]=key2
        r=3.14
        for i in range(1,sz):
            x[i]=x[i-1]*r*(1-x[i-1])
            #print(x[i])
            y[i]=int((x[i]*10**5)%255)
        value=np.reshape(y,(1000,1000,3))
        return value
    
    b=np.empty((framecount,5))
    b=np.load('coordinates.npy')
    i=0
    length=framecount
    value=value1()
    while i<length:
        framenumbers=b[i,0]
        r1=b[i,1]
        r2=b[i,2]
        c1=b[i,3]
        c2=b[i,4]
        name = 'frameencc'+ str(framenumbers) + '.bmp'
        name1 = 'framedecc'+ str(framenumbers) + '.bmp'

        #name = 'C:/Users/ISE-HOD/Documents/proj 21-22/batch 1/codes/data/dataenc/frameencc'+ str(framenumbers) + '.bmp'
        #name1 = 'C:/Users/ISE-HOD/Documents/proj 21-22/batch 1/codes/data/datadec/framedecc'+ str(framenumbers) + '.bmp'
    

        #print(name)
        decimg = cv2.imread(name)
        #print(decimg.shape)
        # print(img)
        #print(r1,r2,c1,c2)
        decimg=np.array(decimg)
        #print(value[r1:r2,c1:c2])
        #cv2.imshow('file t',decimg[r1:r2,c1:c2])
        
        decimg[r1:r2,c1:c2,0] = decimg[r1:r2,c1:c2,0]^value[r1:r2,c1:c2,0]
        decimg[r1:r2,c1:c2,1] = decimg[r1:r2,c1:c2,1]^value[r1:r2,c1:c2,1]
        decimg[r1:r2,c1:c2,2] = decimg[r1:r2,c1:c2,2]^value[r1:r2,c1:c2,2]
        #print(decimg[r1:r2,c1:c2])
        
        cv2.imwrite(name1,decimg)
        i=i+1
        #cv2.imshow('file t',decimg)
    img=cv2.imread('framedecc0.bmp')
    print('i=',i)
    #img=cv2.imread('C:/Users/ISE-HOD/Documents/proj 21-22/batch 1/codes/data/datadec/framedecc0.bmp')
    
    #img=cv2.resize(img,(200,200))
    #winname='Decrypted FRAME'
    #cv2.namedWindow(winname)
    #cv2.moveWindow(winname,400,100)
    cv2.imshow('First stage Decryption',img)
    print('First stage decryption over')
    messagebox.showinfo("Msg", "First stage decryption completed")
    decrypt_second_stage()

def decrypt_second_stage():

    key1=int(t2.get())
    #random.seed(seed)
    #framecount=405
    b=np.empty((framecount,5))
    b=np.load('coordinates.npy')
    i=0
    length=framecount
    while i<length:
        framenumbers=b[i,0]
        r1=b[i,1]
        r2=b[i,2]
        c1=b[i,3]
        c2=b[i,4]
        name = 'framedecc'+ str(framenumbers) + '.bmp'
        name1 ='frameorg'+ str(framenumbers) + '.bmp'

        #name = 'C:/Users/ISE-HOD/Documents/proj 21-22/batch 1/codes/data/datadec/framedecc'+ str(framenumbers) + '.bmp'
        #name1 ='C:/Users/ISE-HOD/Documents/proj 21-22/batch 1/codes/data/datadec/frameorg'+ str(framenumbers) + '.bmp'
        
        img=cv2.imread(name)
        #cv2.imshow('1',img)

        rows=r2-r1
        cols=c2-c1
        newimg=img
        np.random.seed(key1)
        colval=np.random.randint(0,255, size=(cols), dtype=np.uint8)
        colind=np.argsort(colval)
        array = np.arange(cols)
        newimg[r1:r2,colind+c1]=img[r1:r2,array+c1]

        cv2.imshow('Row shuffled image',newimg)
        
        finalimg=newimg
        np.random.seed(key1)
        rowval=np.random.randint(0,255, size=(rows), dtype=np.uint8)
        rowind=np.argsort(rowval)
        finalimg[r1+rowind,c1:c2]=newimg[r1+array,c1:c2]
        cv2.imshow('Original image',finalimg)
        
        cv2.imwrite(name1,finalimg)
        i=i+1
    
    messagebox.showinfo("Msg", "Second stage decryption completed")
    

    #cv2.imshow('Row and Column shuffle reversed',finalimg)
    print('Second stage decryption over')



def make_video():
    #print(framecount)
    #print(framewidth)
    #print(frameheight)
    #print(fps)
    width=480
    height=848
    img_array = []
    for filename in glob.glob('frameorg*.bmp'):
    #for filename in glob.glob('C:/Users/ISE-HOD/Documents/proj 21-22/batch 1/codes/data/datadec/frameorg*.bmp'):

        img = cv2.imread(filename)
        #height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)


    out = cv2.VideoWriter('Decvideo.avi',cv2.VideoWriter_fourcc(*'DIVX'), 29, size)
 
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    tkinter.messagebox.showinfo("Msg", "Video make over")
    print('video make over')

def close():
    win.destroy()
    


# Add an optional Label widget
Label1=Label
t1=Label(win, text= "Privacy Preservation In Surveillance Video Using Selective Encryption", font= ('Aerial 25 bold italic')).pack(pady= 30)
Label(win, text= "Enter secret key 1", font= ('calibri',20,'bold'),foreground='blue').place(x=100,y=300)

Label(win, text= "Enter secret key 2", font= ('calibri',20,'bold'),foreground='blue').place(x=100,y=400)
#l1.place(x=400,y=400)

global t2
t2=Entry(win,font="10",show="*")
t2.place(x=320,y=305, width=80, height=30)

global t3
t3=Entry(win,font="10",show="*")
t3.place(x=320,y=400, width=80, height=30)

# Create a Button to display the message
btn1=ttk.Button(win, text= "Select a Video",style='W.TButton',command=openfn)

btn2=ttk.Button(win, text= "Extract Frames",style='W.TButton',command=extract_frames,compound="right")

#btn3=ttk.Button(win, text= "Decrypt",style='W.TButton',command=decrypt_rowvalcolval,compound="right")

btn4=ttk.Button(win, text= "Decrypt",style='W.TButton',command=decrypt_face,compound="right")

btn5=ttk.Button(win, text= "Play video",style='W.TButton',command=make_video)

btn6=ttk.Button(win, text= "Exit",style='W.TButton',command=close)

btn1.place(x=100,y=100)
btn2.place(x=100,y=200)
#btn3.place(x=100,y=300)
btn4.place(x=100,y=500)
btn5.place(x=100,y=600)
btn6.place(x=970,y=600)
win.mainloop()

