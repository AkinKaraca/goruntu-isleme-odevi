#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#ifdef ENABLE_CUDA
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#endif
#include <algorithm>
#include <atomic>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

cv::Mat read_image(const fs::path& file) {
    std::ifstream input(file, std::ios::binary);
    if (!input) return {};
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(input)), {});
    if (bytes.empty()) return {};
    return cv::imdecode(bytes, cv::IMREAD_UNCHANGED);
}

void write_png(const fs::path& file, const cv::Mat& image) {
    std::vector<unsigned char> bytes;
    if (!cv::imencode(".png", image, bytes))
        throw std::runtime_error("PNG kodlanamadi");
    std::ofstream output(file, std::ios::binary);
    if (!output) throw std::runtime_error("Cikti dosyasi acilamadi");
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    output.close();
    if (!output) throw std::runtime_error("Cikti yazilamadi");
}

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::cerr << "Kullanim: boyutlandir GIRDI_DIZINI CIKTI_DIZINI [--cuda] [--threads N]\n";
            return 1;
        }
        bool use_cuda = false;
        int threads = 1;
        for (int i = 3; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--cuda") use_cuda = true;
            else if (arg == "--threads" && i + 1 < argc) {
                const std::string value = argv[++i];
                std::size_t used = 0;
                threads = std::stoi(value, &used);
                if (used != value.size() || threads < 1 || threads > 64)
                    throw std::runtime_error("Is parcacigi sayisi 1..64 olmali");
            } else throw std::runtime_error("Bilinmeyen veya eksik arguman: " + arg);
        }
        const fs::path input = fs::u8path(argv[1]);
        const fs::path output = fs::u8path(argv[2]);
        if (!fs::is_directory(input)) throw std::runtime_error("Girdi dizini bulunamadi");
        if (fs::weakly_canonical(input) == fs::weakly_canonical(output))
            throw std::runtime_error("Girdi ve cikti dizinleri farkli olmali");
        if (use_cuda) {
#ifdef ENABLE_CUDA
            if (cv::cuda::getCudaEnabledDeviceCount() < 1)
                throw std::runtime_error("Kullanilabilir CUDA GPU yok");
            if (!cv::cuda::DeviceInfo(0).isCompatible())
                throw std::runtime_error("OpenCV derlemesi GPU mimarisi ile uyumsuz");
            cv::cuda::setDevice(0);
            cv::cuda::printShortCudaDeviceInfo(0);
#else
            throw std::runtime_error("Program CUDA destegiyle derlenmemis");
#endif
        }
        const std::set<std::string> extensions = {
            ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".pgm", ".ppm"
        };
        std::vector<fs::path> files;
        for (const auto& entry : fs::directory_iterator(input)) {
            if (!entry.is_regular_file()) continue;
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (extensions.count(ext)) files.push_back(entry.path());
        }
        std::sort(files.begin(), files.end());
        if (files.empty()) throw std::runtime_error("Dizinde desteklenen goruntu dosyasi yok");
        fs::create_directories(output);
        cv::setNumThreads(1);
        std::atomic<std::size_t> next{0}, done{0}, skipped{0}, failed{0};
        std::mutex log_mutex;
        auto worker = [&]() {
            while (true) {
                const auto index = next.fetch_add(1);
                if (index >= files.size()) break;
                const auto& file = files[index];
                try {
                    const cv::Mat source = read_image(file);
                    if (source.empty()) {
                        ++skipped;
                        std::lock_guard<std::mutex> lock(log_mutex);
                        std::cerr << "ATLANDI: " << file.filename().u8string() << '\n';
                        continue;
                    }
                    if ((source.depth() != CV_8U && source.depth() != CV_16U) ||
                        (source.channels() != 1 && source.channels() != 3 && source.channels() != 4))
                        throw std::runtime_error("PNG icin 8/16 bit ve 1/3/4 kanal gerekli");
                    const fs::path target = output / fs::u8path(file.filename().u8string() + ".png");
                    if (fs::exists(target)) throw std::runtime_error("Cikti zaten var; farkli dizin secin");
                    cv::Mat resized;
                    if (use_cuda) {
#ifdef ENABLE_CUDA
                        cv::cuda::setDevice(0);
                        cv::cuda::GpuMat src_gpu, dst_gpu;
                        src_gpu.upload(source);
                        cv::cuda::resize(src_gpu, dst_gpu, cv::Size(1024, 768), 0, 0, cv::INTER_LINEAR);
                        dst_gpu.download(resized);
#endif
                    } else {
                        cv::resize(source, resized, cv::Size(1024, 768), 0, 0, cv::INTER_LINEAR);
                    }
                    write_png(target, resized);
                    ++done;
                    std::lock_guard<std::mutex> lock(log_mutex);
                    std::cout << file.filename().u8string() << " : " << source.cols << 'x'
                              << source.rows << " -> 1024x768 : " << target.filename().u8string() << '\n';
                } catch (const std::exception& e) {
                    ++failed;
                    std::lock_guard<std::mutex> lock(log_mutex);
                    std::cerr << "HATA: " << file.filename().u8string() << " : " << e.what() << '\n';
                }
            }
        };
        std::vector<std::thread> pool;
        const auto count = std::min<std::size_t>(static_cast<std::size_t>(threads), files.size());
        try {
            for (std::size_t i = 0; i < count; ++i) pool.emplace_back(worker);
        } catch (...) {
            for (auto& thread : pool) thread.join();
            throw;
        }
        for (auto& thread : pool) thread.join();
        std::cout << "TAMAM: " << done << " ATLANDI: " << skipped << " HATA: " << failed << '\n';
        return failed ? 3 : (done ? 0 : 2);
    } catch (const std::exception& e) {
        std::cerr << "HATA: " << e.what() << '\n';
        return 1;
    }
}
