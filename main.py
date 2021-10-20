import numpy as np
from scipy.io import wavfile
import cv2 as cv
import os
import math
width = 1920//2
height = 1080//2

# This sets some variables, while the contents of song.wav are never used
rate, amps = wavfile.read("song.wav")
print(rate, amps.max())
secconds = (len(amps) / rate)
print(secconds)
hzPF = rate * (1/30)
print(rate * (1/60))
print(amps.shape)
# We assume a frame rate of 60 fps
out = cv.VideoWriter('outpy.avi',cv.VideoWriter_fourcc('M','J','P','G'), 60, (width, height))

# This makes a list of all files to be used in the visualisation
tracks = os.listdir("mono32fl/")
print(tracks)

# Sets the offset for the different track visualisations
offset = 60
offsets = []
for x in range(len(tracks)):
    offsets.append(offset * x - ((len(tracks) // 2) * offset))
print(offsets)

# Sets up more values for the visualisation, including reading wach wav file into a list of lists
amps = []
rate = 0
totalTime = 0
for f in tracks:
    path = "mono32fl/" + f
    print(path)
    rate, a = wavfile.read(path)
    if(len(a) > 30 and a.max() > 0):
        amps.append(a)
        if len(a) > totalTime:
            totalTime = len(a)
    print(totalTime)
# Sets the framerate of the video
herzPerFrame = rate * 1/60
print(herzPerFrame)

# Loads two style images to be used for making the visualisation
grad = cv.imread("gradient.png")
bg = cv.imread("gradientBG.png")
cv.imshow("image", grad)
cv.waitKey(0)
cv.imshow("image", bg)
cv.waitKey(0)

# This makes a list of lists with one list for each wav file
total = [0] * len(amps)
# This loops for every sample (Sample rate of 44100 hrz)
for x in range(totalTime):
    # For each frame we create a new image
    if x % herzPerFrame == 0 and x != 0:
        image = np.zeros((height, width), np.uint8)
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    # This loops for each track
    for y in range(len(amps)):
        try:
            # Since our sample rate is much larger than our fps we will average all amplitude values for an individual track for each frame
            total[y] += abs(amps[y][x])
            # This if statement only runs if the current step of the loop is one we want to make a video frame for
            if x % herzPerFrame == 0 and x != 0:
                # Zeros out all variables
                rec = np.zeros((height, width), np.uint8)
                edge = np.zeros((height, width), np.uint8)
                # Makes the inner rectangle mask, as well as the border mask
                rec = cv.rectangle(rec, (width // 2 - offsets[y], height), ((width // 2) - (offsets[y] + 40), height - int(height * ((total[y]/herzPerFrame)/(amps[y].max())))), (255, 255, 255), -1)
                edge = cv.rectangle(edge, (width // 2 - offsets[y], height), ((width // 2) - (offsets[y] + 40), height - int(height * ((total[y]/herzPerFrame)/(amps[y].max())))), (255, 255, 255), 3)
                # This isn't nececary, but because of superstition i'm keeping it in
                image = cv.rectangle(image, (width // 2 - offsets[y], height), ((width // 2) - (offsets[y] + 40), height - int(height * ((total[y]/herzPerFrame)/(amps[y].max())))), (255, 255, 255), -1)

                # This just checks to see if the average amplitude is zero, as we wouldn't need to desplay anything, also dividing by zero will crash the program
                if ((total[y]/herzPerFrame)) != 0:
                    # This loads in the fill and border images and resizes them acordingly
                    tempGrad = cv.resize(cv.imread("gradient.png"), (width, height - (height - int(math.ceil(height * ((total[y]/herzPerFrame)/(amps[y].max())))))))
                    tempBG = cv.resize(cv.imread("gradientBG.png"), (width, height - (height - int(math.ceil(height * ((total[y]/herzPerFrame)/(amps[y].max())))))))
                    newGrad = np.zeros((height, width), np.uint8)
                    newBG = np.zeros((height, width), np.uint8)
                    newGrad = cv.cvtColor(newGrad, cv.COLOR_GRAY2BGR)
                    newBG = cv.cvtColor(newBG, cv.COLOR_GRAY2BGR)
                    newGrad[height - np.shape(tempGrad)[0]:height, 0:width] = tempGrad
                    newBG[height - np.shape(tempBG)[0]:height, 0:width] = tempBG

                    # Masks the fill and border with the main image
                    forFrame = cv.bitwise_or(newGrad, newGrad, mask = rec)
                    rec = cv.bitwise_not(rec)
                    backFrame = cv.bitwise_or(image, image, mask = rec)
                    image = cv.bitwise_or(forFrame, backFrame)

                    forFrame = cv.bitwise_or(newBG, newBG, mask = edge)
                    edge = cv.bitwise_not(edge)
                    backFrame = cv.bitwise_or(image, image, mask = edge)
                    image = cv.bitwise_or(forFrame, backFrame)
                

        # Some wav file samples are shorter than others
        except IndexError as e:
            print(e)
    # This takes the finished image and writes it to the video file
    if x % herzPerFrame == 0 and x != 0:
        # This resets the averaging   
        total = [0] * len(amps)
        
        out.write(image)      
        cv.imshow("image", image)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
         
out.release()
print("done")  
cv.destroyAllWindows()