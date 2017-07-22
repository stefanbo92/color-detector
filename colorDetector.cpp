#include "opencv2/opencv.hpp"
#include <iostream>
#include <sys/types.h>
#include <dirent.h>

using namespace std;
using namespace cv;

//PARAMETER CONFIGURATION
bool useClahe=true;  //use CLAHE
int clipLimit=5; //clip limit of CLAHE
bool useHSV=true; //use HSV or RGB color space
int histSize = 50; //histogram bin size per channel
Size imgSize=Size(250,250); //input image size
double lx= 0.105 ; //ROI crop: left margin
double ly= 0.30 ; //ROI crop: top margin
double rx= 0.105 ; //ROI crop: right margin
double ry= 0.2 ; //ROI crop: bottom margin
int gaussianBlurSize=-1; // size of guassian blur filter kernel
bool useHazeRemoval=false; //apply haze removal
bool useHistEq=false; //use Histogram Equalisation
bool normalizeHist=false; //normalize Histograms
bool zeroMeanUnitVar=true; //make feature vector zero mean and unit variance
vector<string> classes; //vector holding the names of classes


// fill in the classes you want to use, there should be one folder for each class in the data folder
void initClasses(){
    classes.push_back("blue");
    classes.push_back("red");
    classes.push_back("white");
    classes.push_back("black");
    classes.push_back("gray");
}

//calculate three channel histograms for an image
vector<Mat> calcHists(Mat img){
    vector<Mat> hists;
    Mat histMask=Mat();

    // Separate the image in 3 places ( B, G and R )
    vector<Mat> bgr_planes;
    split( img, bgr_planes );

    // Set the ranges ( for B,G,R) )
    float range[] = { 0, 256 } ;
    const float* histRange = { range };

    bool uniform = true; bool accumulate = false;

    Mat b_hist, g_hist, r_hist;

    // Compute the histograms:
    calcHist( &bgr_planes[0], 1, 0, histMask, b_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes[1], 1, 0, histMask, g_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes[2], 1, 0, histMask, r_hist, 1, &histSize, &histRange, uniform, accumulate );

    hists.push_back(b_hist);
    hists.push_back(g_hist);
    hists.push_back(r_hist);

    return hists;
}

//apply CLAHE to image
void clahe(Mat& img){
    // convert the RGB color image to Lab
    cvtColor(img, img, CV_BGR2Lab);

    // Extract the L channel
    vector<Mat> lab_planes(3);
    split(img, lab_planes);  // now we have the L image in lab_planes[0]

    // apply the CLAHE algorithm to the L channel
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(clipLimit);
    cv::Mat dst;
    clahe->apply(lab_planes[0], dst);

    // Merge the the color planes back into an Lab image
    dst.copyTo(lab_planes[0]);
    cv::merge(lab_planes, img);

    // convert back to RGB
    cv::cvtColor(img, img, CV_Lab2BGR);
}

//load and preprocess image
Mat preprocessImg(Mat src){
    resize(src, src, imgSize,0,0,INTER_CUBIC);

    //gaussian blur
    if(gaussianBlurSize>0){
        GaussianBlur( src, src, Size( gaussianBlurSize, gaussianBlurSize ), 0, 0 );
    }

    //apply CLAHE
    if(useClahe){
        clahe(src);
    }

    //crop image
    src = src(Rect(imgSize.width*lx, imgSize.height*ly, imgSize.width*(1-lx-rx), imgSize.height*(1-ly-ry)));

    //convert to HSV
    if (useHSV) cvtColor(src, src, CV_BGR2HSV);

    return src;
}

//helper function to get label according to classNumber
string getLabel(int classNo){
    string label=classes[classNo];
    return label;
}

//get histogram of image and reshape to single row vector
Mat getHistReshape(Mat src){
    //load and preprocess image
    Mat testImg=preprocessImg(src);
    //calculate histograms
    vector<Mat> hists=calcHists(testImg);
    //reshape histograms
    Mat histsMat;
    histsMat.push_back(hists[0]);
    histsMat.push_back(hists[1]);
    histsMat.push_back(hists[2]);
    //normalize histograms
    if (normalizeHist) normalize(histsMat, histsMat, 0, 1, NORM_MINMAX, -1, Mat() );
    histsMat=histsMat.reshape(1,1);
    histsMat.convertTo(histsMat, CV_32F);
    if(zeroMeanUnitVar) { //make feature vector zero mean and unit variance
        Mat mean,stddev;
        meanStdDev(histsMat,  mean,  stddev);
        histsMat=(histsMat-mean)/stddev;
    }
    return histsMat;
}

//train SVM with training data
void trainSVM(vector<string> dirs,CvSVM& SVM){
    Mat labels;
    Mat trainingData;
    cout<<"loading images..."<<endl;
    //loop over all directories
    for(int i=0;i<dirs.size();i++){
        string dir=dirs[i];
        //open directory
        DIR *dp;
        struct dirent *dirp;
        if((dp  = opendir(dir.c_str())) == NULL) {
            cout << "Error, cant find directory!" << endl;
        }
        //loop through all files in that directory
        while ((dirp = readdir(dp)) != NULL) {
            string name=string(dirp->d_name);
            if(name!="."&&name!=".."){
                //read image and preprocess it
                Mat img=imread(dir+name);
                img=getHistReshape(img);
                // Set up training data
                labels.push_back(i);
                trainingData.push_back(img);
            }
        }
        closedir(dp);
    }

    // convert data to be 32 bit floating numbers
    labels.convertTo(labels, CV_32F);
    trainingData.convertTo(trainingData, CV_32F);

    //print size of data and labels
    cout<<"label size: "<<labels.size()<<endl;
    cout<<"training Data size: "<<trainingData.size()<<endl;

    // Set up SVM's parameters
    CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::POLY;
    params.degree=2;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 10000, 1e-9);

    // Train the SVM
    cout<<"training SVM..."<<endl;
    SVM.train(trainingData, labels, Mat(), Mat(), params);
}

//predic color for test image
int testSVM(Mat testImg,CvSVM& SVM){
    float response;
    testImg=getHistReshape(testImg);
    response = SVM.predict(testImg);
    return (int)response;
}
