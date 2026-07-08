//  Internal C++/OpenCV helpers shared by the Obj-C++ vision units.
//  Only ever included from .mm translation units (pulls in the OpenCV C++ API).
#pragma once

#import <CoreVideo/CoreVideo.h>
#import <opencv2/opencv.hpp>

namespace rc {

/// Wrap a 32BGRA CVPixelBuffer as a 3-channel BGR cv::Mat (deep copy, so the
/// caller can unlock the buffer immediately). Matches the BGR frames PyAV feeds
/// the desktop pipeline.
inline cv::Mat bgrFromPixelBuffer(CVPixelBufferRef pb) {
    CVPixelBufferLockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
    const int w = (int)CVPixelBufferGetWidth(pb);
    const int h = (int)CVPixelBufferGetHeight(pb);
    const size_t stride = CVPixelBufferGetBytesPerRow(pb);
    void *base = CVPixelBufferGetBaseAddress(pb);
    cv::Mat bgra((int)h, (int)w, CV_8UC4, base, stride);
    cv::Mat bgr;
    cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);
    CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
    return bgr;   // owns its own storage
}

/// Letterbox geometry outputs.
struct LetterboxGeom { float ratio; int padLeft; int padTop; };

/// Core letterbox shared by the exact (static) and stride-32 (dynamic) paths.
/// Faithful to `yolo_onnx_runner.letterbox` / `letterbox_exact`:
/// INTER_LINEAR resize, center pad with (114,114,114), asymmetric rounding.
inline cv::Mat letterbox(const cv::Mat &img, int targetW, int targetH,
                         bool exact, int stride, LetterboxGeom &geom) {
    const int h = img.rows, w = img.cols;
    float r;
    int newUnpadW, newUnpadH;
    float dw, dh;
    if (exact) {
        r = std::min((float)targetH / h, (float)targetW / w);
        newUnpadW = std::min(targetW, (int)std::lround(w * r));
        newUnpadH = std::min(targetH, (int)std::lround(h * r));
        dw = (targetW - newUnpadW) / 2.0f;
        dh = (targetH - newUnpadH) / 2.0f;
    } else {
        // targetW == targetH == imgsz here.
        r = std::min((float)targetH / h, (float)targetW / w);
        newUnpadW = (int)std::lround(w * r);
        newUnpadH = (int)std::lround(h * r);
        dw = ((targetW - newUnpadW) % stride) / 2.0f;
        dh = ((targetH - newUnpadH) % stride) / 2.0f;
    }
    cv::Mat resized;
    if (w != newUnpadW || h != newUnpadH) {
        cv::resize(img, resized, cv::Size(newUnpadW, newUnpadH), 0, 0, cv::INTER_LINEAR);
    } else {
        resized = img;
    }
    const int top = (int)std::lround(dh - 0.1f);
    const int bottom = (int)std::lround(dh + 0.1f);
    const int left = (int)std::lround(dw - 0.1f);
    const int right = (int)std::lround(dw + 0.1f);
    cv::Mat out;
    cv::copyMakeBorder(resized, out, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    geom.ratio = r;
    geom.padLeft = left;
    geom.padTop = top;
    return out;   // BGR, size (newUnpad + pad)
}

/// Pack a BGR letterboxed Mat into a float32 NCHW RGB /255 tensor.
/// Mirrors `img[..., ::-1].transpose(2,0,1)[None] / 255.0`.
inline void packTensorRGB(const cv::Mat &bgr, float *out) {
    const int H = bgr.rows, W = bgr.cols, plane = H * W;
    for (int y = 0; y < H; ++y) {
        const cv::Vec3b *row = bgr.ptr<cv::Vec3b>(y);
        for (int x = 0; x < W; ++x) {
            const cv::Vec3b &px = row[x];       // B,G,R
            const int i = y * W + x;
            out[i] = px[2] / 255.0f;            // R plane
            out[plane + i] = px[1] / 255.0f;    // G plane
            out[2 * plane + i] = px[0] / 255.0f;// B plane
        }
    }
}

} // namespace rc
