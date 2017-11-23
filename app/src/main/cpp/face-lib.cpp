

#include <jni.h>
#include <Eigen/Dense> // http://eigen.tuxfamily.org
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <FaceRecognitionLib/Eigenfaces.h>
#include <FaceRecognitionLib/Fisherfaces.h>
#include <FaceRecognitionLib/Tools.h>
#include <android/log.h>

#ifdef NDEBUG
#define LOGD(...) ((void)0)
#define LOGI(...) ((void)0)
#define LOGE(...) ((void)0)
#define LOG_ASSERT(condition, ...) ((void)0)
#else
#define LOG_TAG "FaceRecognitionAppActivity/Native"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))
#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__))
#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__))
#define LOG_ASSERT(condition, ...) if (!(condition)) __android_log_assert(#condition, LOG_TAG, __VA_ARGS__)
#endif

Eigenfaces eigenfaces;
Fisherfaces fisherfaces;

using namespace std;
using namespace cv;
using namespace Eigen;

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT void JNICALL Java_com_lauszus_facerecognitionapp_NativeMethods_TrainFaces(JNIEnv, jobject, jlong addrImages, jlong addrClasses) {
    Mat *pImages = (Mat *) addrImages; // Each images is represented as a column vector
    Mat *pClasses = (Mat *) addrClasses; // Classes are represented as a vector

    LOG_ASSERT(pImages->type() == CV_8U, "Images must be an 8-bit matrix");
    MatrixXi images;
    cv2eigen(*pImages, images); // Copy from OpenCV Mat to Eigen matrix

    //Facebase *pFacebase;
    if (pClasses == NULL) { // If classes are NULL, then train Eigenfaces
        eigenfaces.train(images); // Train Eigenfaces
        LOGI("Eigenfacess numComponents: %d", eigenfaces.numComponents);
        //pFacebase = &eigenfaces;
    } else {
        LOG_ASSERT(pClasses->type() == CV_32S && pClasses->cols == 1, "Classes must be a signed 32-bit vector");
        VectorXi classes;
        cv2eigen(*pClasses, classes); // Copy from OpenCV Mat to Eigen vector
        LOG_ASSERT(classes.minCoeff() == 1, "Minimum value in the list must be 1");
        fisherfaces.train(images, classes); // Train Fisherfaces
        LOGI("Fisherfaces numComponents: %d", fisherfaces.numComponents);
        //pFacebase = &fisherfaces;
    }

    /*
    if (!pFacebase->V.hasNaN()) {
        for (int i = 0; i < pFacebase->numComponents; i++) { // Loop through eigenvectors
            for (int j = 0; j < 10; j++) // Print first 10 values
                LOGI("Eigenvector[%d]: %f", i, pFacebase->V(j, i));
        }
    } else
        LOGE("Eigenvectors are not valid!");
    */
}

JNIEXPORT void JNICALL Java_com_lauszus_facerecognitionapp_NativeMethods_MeasureDist(JNIEnv *env, jobject, jlong addrImage, jfloatArray minDist, jintArray minDistIndex, jfloatArray faceDist, jboolean useEigenfaces) {
    Facebase *pFacebase;
    if (useEigenfaces) {
        LOGI("Using Eigenfaces");
        pFacebase = &eigenfaces;
    } else {
        LOGI("Using Fisherfaces");
        pFacebase = &fisherfaces;
    }

    if (pFacebase->V.any()) { // Make sure that the eigenvector has been calculated
        Mat *pImage = (Mat *) addrImage; // Image is represented as a column vector

        VectorXi image;
        cv2eigen(*pImage, image); // Convert from OpenCV Mat to Eigen matrix

        LOGI("Project faces");
        VectorXf W = pFacebase->project(image); // Project onto subspace
        LOGI("Reconstructing faces");
        VectorXf face = pFacebase->reconstructFace(W);

        LOGI("Calculate normalized Euclidean distance");
        jfloat dist_face = pFacebase->euclideanDistFace(image, face);
        LOGI("Face distance: %f", dist_face);
        env->SetFloatArrayRegion(faceDist, 0, 1, &dist_face);

        VectorXf dist = pFacebase->euclideanDist(W);

        vector<size_t> sortedIdx = sortIndexes(dist);
        for (auto idx : sortedIdx)
            LOGI("dist[%zu]: %f", idx, dist(idx));

        int minIndex = (int) sortedIdx[0];
        env->SetFloatArrayRegion(minDist, 0, 1, &dist(minIndex));
        env->SetIntArrayRegion(minDistIndex, 0, 1, &minIndex);
    }
}

#ifdef __cplusplus
}
#endif
