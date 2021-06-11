# Indian Sign Language Detection
* Provided a person makes the gestures in real-time or by feeding an image to the code, the aim is to detect which number he/she wants to convey.
* Here is the approach by which I implemented this task.
![](https://i.imgur.com/X2eZd58.png)
---
## METHOD
1. Contour Detection and Convex hull are the prime features used for implementation.
2. Convex hull is the smallest polygon that encloses all the given points.
3. As mentioned above, the convex hull encloses the given points. These vertices, in the case of a hand, are the fingertips.
4. The coordinates of the following help to calculate the angle.
5. The number of angles found between the tips is one less than the number of fingers.
6. Thresholds imposed help us to avoid some undesired values of angles getting included in the count.
7. Finally, with their number and signs, it is possible to identify the gestures.
---
### APPLICATION
Can be used to teach this language to deaf and dumb people.
One can identify various gestures for hands-free usage thereby being user-friendly.

#### [RESULTS](https://drive.google.com/drive/folders/1Jmot1vzmK7iah3CArOsAG7uXCEgHNveV?usp=sharing)
Here is the drive link for the results


SOFTWARES AND MODULES USED:
Python 3.7
Numpy 1.19.0
OpenCV 4.2.0 


