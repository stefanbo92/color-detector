# SVM Color Detector
This is an OpenCV program that can detect the color of objects in an image. It was part of a school project where we had classify the color of cars in the streets, but you can also train it on your own dataset.

### Installation:
You can build this project using CMake by the following commands:
```bash
mkdir build
cd build
cmake ..
make
```
This will compile the code into the build folder. The programs 'train' and 'test' can be simply run by
```bash
./test
```
The 'test' app will load a pretrained SVM modell ("modell.xml") which is already trained on identifying the color of a car. It will load a test image ("data/test.jpg") and classifies the color. Due to copyright reasons I cannot publish the training data set unfortunately.
### Training your own model:
If you want to train your own model for detecting the color of an object you need to do two steps:
First, you need to adjust the initClasses() method in "colorDetector.cpp". Here you need to put the names of the colors you want to train (e.g. "blue, "black"... etc.).
Second, you need to put your training images into the folder data with one subfolder per class. 

Then you can train the SVM by running
```bash
./train
```
which will save a "modell.xml". You can adjust many parameters of the algorithm and the SVM training in "colorDetector.cpp".

### Algorithm:
Here you can find a short overview of how the color classification algorithm is working:
- loading and resizing the image
- applying contrast limited adaptive histogram equalisation (CLAHE) to the image. The yields better contrast on various illuminations
- cropping a ROI (middle part of the image) to reduce incfluence of the background
- color space conversion to HSV (colors are better to distinguish in this color space)
- calculation of histogram for each color channel 
- reshaping three channel histogram to onedimensional vector
- making the feature vector zero mean and unit variance
- feeding the feature vector into SVM to classify color
- the SVM was trained using a Polynomial kernel of degree two

If you have any questions feel free to write me a message :)
