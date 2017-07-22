#include "colorDetector.cpp"

using namespace std;
using namespace cv;


//main function
int main()
{
    // loading train data
    initClasses();
    vector<string> dirsTrain;
    for (int i=0;i<classes.size();i++){
        dirsTrain.push_back("../data/"+classes[i]+"/");
    }

    // training SVM
    CvSVM SVM;
    trainSVM(dirsTrain,SVM);

    // Save SVM
    cout<<"saving SVM model under "<<"../modell.xml"<<endl;
    SVM.save("../modell.xml");

    waitKey(1);
    return 0;
}
