import numpy as np
import cv2
import os
import time
import copy




block_size = 56
min_block_size = 4
fps = 24

def pre_process_image(img, block_size,min_block_size):
    size = np.asarray(img).shape
    new_i = copy.deepcopy(img)
    if not block_size <= min_block_size:
        for fila in range(0,size[0],block_size):
            for columna in range(0,size[1],block_size):
                if len(size) > 2:
                    new_i[fila:fila+block_size,columna:columna+block_size,0] = np.mean(img[fila:fila+block_size,columna:columna+block_size,0])
                    new_i[fila:fila+block_size,columna:columna+block_size,1] = np.mean(img[fila:fila+block_size,columna:columna+block_size,1])
                    new_i[fila:fila+block_size,columna:columna+block_size,2] = np.mean(img[fila:fila+block_size,columna:columna+block_size,2])
                else:
                    new_i[fila:fila+block_size,columna:columna+block_size] = np.mean(img[fila:fila+block_size,columna:columna+block_size])
        
    return new_i
   
def recursivitat_diver(mask,frame,max_block_size,min_block_size):
    size = np.asarray(frame).shape
    new_i = copy.deepcopy(np.asarray(frame))
    
    for fila in range(0,size[0],max_block_size):
        for columna in range(0,size[1],max_block_size):
            porcio_of_pixels_in_mask = np.mean(mask[fila:fila+max_block_size,columna:columna+max_block_size])/255 #la media / el valor que toman si són parte de la mascara
            number_of_pixels_out = np.square(max_block_size)*porcio_of_pixels_in_mask # proporcion de pixeles en la mascara * numero de pixeles en la mascara 
            # fer un bloc de: tamany del bloc màxim (max_block_size) * porció de pixels que no son background
            new_i[fila:fila+max_block_size,columna:columna+max_block_size,:]=pre_process_image(frame[fila:fila+max_block_size,columna:columna+max_block_size,:],int(max_block_size*porcio_of_pixels_in_mask),min_block_size)

    return new_i



if __name__ == "__main__":
    folder_path = 'C:/Users/Carles/Documents/Code/malnascudes/glitch/captures'
    default_background_folder = 'C:/Users/Carles/Documents/Code/malnascudes/glitch/background'
    default_background = 'capture_num3.png'

    start_time = time.time()

    cap = cv2.VideoCapture(1)

    ret, ref_frame = cap.read()
    ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
    #load first backgrund
    backSub = cv2.createBackgroundSubtractorKNN()
    backSub.setHistory(15)

    threshold = 33
    background_learned = False
    #for image processing, erode etc etc
    kernel_e = np.ones((5,5),np.uint8)
    kernel_c = np.ones((5,5),np.uint8)
    kernel_f = np.ones((5,5),np.float32)/25
    #gray_bckg = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    while(True):    
        # handle key actions, if key == q True is returned to end loop, otherwise False is returned
        ######################################################################################################################
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            image_name='capture_num'+ str(len([n for n in os.listdir(folder_path) if n.endswith('.png')]))+'.png'
            cv2.imwrite(os.path.join(folder_path,image_name),frame) 
            print(image_name + ' saved into ' + folder_path)
        elif key == ord('b'):
            background = frame
            print("New background taken")
        
        elif key == ord('+'):
            block_size += 1
            print("New block size = " + str(block_size))
        elif key == ord('-'):
            block_size -= 1
            print("New block size = " + str(block_size))

        ######################################################################################################################
        if (time.time()-start_time)>(1/fps):
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Display the resulting frame

            fgMask = backSub.apply(frame)
            glitched = recursivitat_diver(fgMask,frame,block_size,min_block_size)
            cv2.imshow('pixeeeeel',glitched)
            cv2.imshow('mask',fgMask)
            
            print(str(1/(time.time()-start_time)) + " fps")
            start_time = time.time()
            ref_frame = frame
            ref_gray = gray



    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

