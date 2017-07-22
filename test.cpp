#include "opencv2/opencv.hpp"
#include <iostream>
#include <ctime>
#include <sys/types.h>
#include <dirent.h>

#include "colorDetector.cpp"

using namespace std;
using namespace cv;


//main function
int main()
{
    // loading SVM model and classifying color on testimage
    initClasses();
    CvSVM SVM;
    SVM.load("../modell.xml");

    string imagename="../data/test.jpg";
    Mat testImg=imread(imagename);
    int response=testSVM(testImg,SVM);

    cout<<"image "<<imagename<<" is classified as "<<getLabel(response)<<endl;

    waitKey(1);
    return 0;
}
