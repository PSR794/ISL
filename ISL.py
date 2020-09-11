import cv2 as cv
import numpy as np
import math
#declaring varibles
font = cv.FONT_HERSHEY_SIMPLEX
kernel=np.ones((5,5),np.uint8)
vid=cv.VideoCapture(0)
if not vid.isOpened():
    print('camera didnt open')
    exit()
Area=[]
AREA=[]
k=0
print('MAKE A FIVE WITH YOUR WHICH TO BE USED AND PRESS E')
#TO detect the hand (whether left or right)
while True:
    Return,Frame=vid.read()
    HSV=cv.cvtColor(Frame,cv.COLOR_BGR2HSV)
    mask=cv.inRange(HSV,np.array([0,0,22]),np.array([28,249,254]))
    HA=Frame[93:358,153:428]
    HAND_AREA_M=mask[93:358,153:428]
    CANNY_OP=cv.dilate(HAND_AREA_M,kernel,iterations=1)
    HAND=cv.bitwise_and(HA,HA,mask=CANNY_OP)
    cv.rectangle(Frame,(150,90),(430,360),(0,255,255),2)
    HAND1=HAND.copy()
    
    canny=cv.Canny(CANNY_OP,100,200)
    canny[canny.shape[0]-6:canny.shape[0],:]=0
    cnt,h=cv.findContours(canny,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    cv.imshow('original',Frame)
    cv.imshow('canny',canny)
    
    if (cv.waitKey(3) & 0xFF==ord('e')):
        break
vid.release()
cv.destroyAllWindows()
canny=cv.Canny(CANNY_OP,100,200)
canny[canny.shape[0]-6:canny.shape[0],:]=0
cont,h=cv.findContours(canny,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
cmax = max(cont, key = cv.contourArea)

# calculating extreme points 

left=list(cmax[cmax[:,:,0].argmin()][0])
right=list(cmax[cmax[:,:,0].argmax()][0])
print(left,right)



#capturing and reading frames for main execution
vid=cv.VideoCapture(0)
if not vid.isOpened():
    print('camera didnt open')
    exit()
    
# getting the angle value for the condition
hand_side=(np.arctan((right[1]-left[1])/(right[0]-left[0]))*180)/np.pi
print('make gestures in the box,when finished press E')
while True:
    Return,Frame=vid.read()
    #flipping the hand if its a right one
    if hand_side>0:
        Frame=cv.flip(Frame,1)
        
    # creating the mask
    HSV=cv.cvtColor(Frame,cv.COLOR_BGR2HSV)
    mask=cv.inRange(HSV,np.array([0,0,22]),np.array([28,249,254]))
    HAND_AREA=Frame[93:278,153:368]
    HAND_AREA_M=mask[93:278,153:368]
    CANNY_OP=cv.dilate(HAND_AREA_M,kernel,iterations=1)
    HAND=cv.bitwise_and(HAND_AREA,HAND_AREA,mask=CANNY_OP)
    cv.rectangle(Frame,(150,90),(370,280),(0,255,255),2)
#    H=HAND[53:378,103:448]
    HAND1=HAND.copy()

    # edgw image of the hand
    canny=cv.Canny(CANNY_OP,100,200)
    canny[canny.shape[0]-6:canny.shape[0],:]=0
    cnt,h=cv.findContours(canny,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    if np.size(cnt)==0:
        continue
    han=cnt[0]
#    hull=cv.convexHull(hand)
    hull=[]
    MA=0
    draw=[]
    index=0
    l=0
    ################################
    #getting the edge image.
    CANNY_D1=cv.GaussianBlur(CANNY_OP,(5,5),100)
    cnt,h=cv.findContours(CANNY_D1,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    cmax = max(cnt, key = cv.contourArea)


    #finding the convex hull
    hull1=cv.convexHull(cmax,returnPoints=False)
#    if np.size(hulll)==0:
#        continue
    defects = cv.convexityDefects(cmax, hull1)
    if np.size(defects)==0:
        continue

    #finding convexity defects
    if(str(type(defects))!="<class 'NoneType'>"):
        for i in range(defects.shape[0]):
            
             s,e,f,d = defects[i,0]
             start = tuple(cmax[s][0])
             end = tuple(cmax[e][0])
             far = tuple(cmax[f][0])
             #cv.line(Frame,start,end,[0,255,0],2)
             #cv.circle(Frame,far,5,[0,0,255],-1)

             x=math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
             y=math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
             z=math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
             s = (x+y+z)/2
             ar = math.sqrt(s*(s-x)*(s-y)*(s-z))
             d=(2*ar)/x
             angle = math.acos((y**2 + z**2 - x**2)/(2*y*z)) * 57
             if angle <= 90 and d>30:
                    l += 1
                    cv.circle(Frame,far, 3, [255,0,0], -1)

        #cv.line(Frame,start, end, [0,255,0], 2)       
    #counting the defects
    l+=1

    
    #fetching the maximum contour area
    for i in range(len(cnt)):
        area=cv.contourArea(cnt[i])
        if area>MA:
            MA=area
            Area.append(area)
            index=i
    hull.append(cv.convexHull(cnt[index]))
    cv.drawContours(HAND,hull,-1,(0,255,255),2)

    if np.size(hull)==0:
        continue
    AREA=[]


    # declaring the bins for selecting fewer points from the cluster
    bins=list(range(0,HAND.shape[1],15))
    if not(HAND.shape[1]%15)==0:
        bins.append(HAND.shape[1])
    Z=hull[0]
    P=[]
    p=[]

    #iterating over to get the angles
    for i in range(1,len(bins)):
        m=HAND.shape[0]
        n=0
        for j in range(len(Z)):
            if Z[j][0][0]>bins[i-1] and Z[j][0][0]<=bins[i]:
                if Z[j][0][1]<m:
                    m=Z[j][0][1]
                    n=Z[j][0][0]
            if (not m==HAND.shape[0]) and (not n==0):
                P.append([n,m])
        
    for i in range(len(P)):
        if P[i][1]<HAND.shape[0]-80:
            cv.circle(HAND1,tuple(P[i]),8,(0,255,0),1)
            p.append(P[i])
            

    # couting permissible angle values
    ANG=[]
    ANG_WC=[]
    for i in range(len(p)-1):
        if not(abs(p[i+1][1]-p[i+1][1])<=17 and abs(p[i][0]-p[i+1][0])<=17):
            v=(p[i+1][1]-p[i][1])/(p[i+1][0]-p[i][0])
            ang=(np.arctan(v)*180)/np.pi
            ANG_WC.append(ang)
#                    print(ang)
    fingers=0

    for i in range(len(ANG_WC)):
        if -75<ANG_WC[i]<55:
            ANG.append(ANG_WC[i])
            fingers+=1
           

    # THE OUTPUT    
    #for 1 (may or may not work)
    if fingers==0:
        cv.putText(Frame,'1',(40,120),font,2,(0,0,255),2,cv.LINE_AA)

    #for 5
    elif l==5 :
        cv.putText(Frame,'5',(40,120),font,2,(0,0,255),2,cv.LINE_AA)

    #for 4
    elif l==4 and np.size(np.where(np.array(ANG)<0))==1:
        cv.putText(Frame,'4',(40,120),font,2,(0,0,255),2,cv.LINE_AA)

    #for 2
    elif l==2 and -30<ANG[0]<0:
        cv.putText(Frame,'2',(40,120),font,2,(0,0,255),2,cv.LINE_AA)

    #9,8,7,3
    elif fingers==2:
    
        if len(ANG_WC)>=2 and l==1:
            cv.putText(Frame,'9',(40,120),font,2,(0,0,255),2,cv.LINE_AA)
        
        elif np.size(np.where(np.array(ANG)<0))==0:
            cv.putText(Frame,'8',(40,120),font,2,(0,0,255),2,cv.LINE_AA)

        elif np.size(np.where(np.array(ANG)>0))==0:
            cv.putText(Frame,'8',(40,120),font,2,(0,0,255),2,cv.LINE_AA)

        elif ANG_WC[0]<-30:
            cv.putText(Frame,'7',(40,120),font,2,(0,0,255),2,cv.LINE_AA)
        else:
            cv.putText(Frame,'3',(40,120),font,2,(0,0,255),2,cv.LINE_AA)

    # for 0
    elif (fingers==4 or fingers==3)and l==1:
        cv.putText(Frame,'0',(40,120),font,2,(0,0,255),2,cv.LINE_AA)

    # for 3-case 2
    elif 1<len(ANG_WC)<4 and l>1:
        if fingers==1 and ANG_WC[1]>0:
            cv.putText(Frame,'3',(40,120),font,2,(0,0,255),2,cv.LINE_AA)

    #for 6  
    elif fingers==1: 
            if -10>ANG_WC[0]>-60: 
                cv.putText(Frame,'6',(40,120),font,2,(0,0,255),2,cv.LINE_AA)

#for 1 (may or may not work)
    elif fingers==1 and (ANG[0]>45 or ANG[0]<(-45)):
        cv.putText(Frame,'1',(40,120),font,2,(0,0,255),2,cv.LINE_AA)

    # for 9 case 2
    else:
        if l==1:
            cv.putText(Frame,'9',(40,120),font,2,(0,0,255),2,cv.LINE_AA)

            
    #cv.drawContours(HAND,hull,-1,(0,255,0),2)
    cv.imshow('original1',Frame)
#    cv.imshow('mask1',mask)
    cv.imshow('hand',HAND)
    cv.imshow('hand1',HAND1)
    cv.imshow('hand2',HAND_AREA)
    cv.imshow('hand3',HAND_AREA_M)
#    cv.imshow('hand4',canny)
#    cv.imshow('hand5',CANNY_OP)    

    
    if (cv.waitKey(3) & 0xFF==ord('e')):
        break
#closing the windows
vid.release()
cv.destroyAllWindows()

