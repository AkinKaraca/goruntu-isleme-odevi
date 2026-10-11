#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <iostream>

int main() {
    try {
        std::cout << cv::getBuildInformation() << '\n';
        const int count = cv::cuda::getCudaEnabledDeviceCount();
        std::cout << "CUDA_DEVICE_COUNT=" << count << '\n';
        if (count < 1 || !cv::cuda::DeviceInfo(0).isCompatible()) return 1;
        cv::cuda::setDevice(0);
        cv::cuda::printShortCudaDeviceInfo(0);
        cv::Mat source(16, 16, CV_8UC1, cv::Scalar(123));
        cv::cuda::GpuMat input, output;
        input.upload(source);
        cv::cuda::resize(input, output, cv::Size(1024, 768), 0, 0, cv::INTER_LINEAR);
        cv::Mat result;
        output.download(result);
        if (result.cols != 1024 || result.rows != 768 || cv::countNonZero(result != 123)) return 2;
        std::cout << "CUDA_RESIZE_TEST=PASS\n";
        return 0;
    } catch (const cv::Exception& e) {
        std::cerr << e.what() << '\n';
        return 3;
    }
}
