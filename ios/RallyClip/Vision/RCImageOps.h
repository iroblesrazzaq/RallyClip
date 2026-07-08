#import <Foundation/Foundation.h>
#import <CoreVideo/CoreVideo.h>

NS_ASSUME_NONNULL_BEGIN

/// Result of a letterbox: the model input tensor (float32 NCHW, RGB, /255) plus
/// the geometry needed to un-letterbox detections back to source pixels.
@interface RCLetterboxResult : NSObject
@property (nonatomic, readonly) NSData *tensor;   // float32, length 3*height*width
@property (nonatomic, readonly) int width;        // letterboxed canvas width
@property (nonatomic, readonly) int height;       // letterboxed canvas height
@property (nonatomic, readonly) float ratio;      // resize scale applied to source
@property (nonatomic, readonly) int padLeft;
@property (nonatomic, readonly) int padTop;
@property (nonatomic, readonly) int origWidth;    // source frame width
@property (nonatomic, readonly) int origHeight;   // source frame height
@end

/// OpenCV image ops shared with the pose runner and court detector, kept
/// pixel-faithful to the desktop cv2 path (`yolo_onnx_runner.letterbox*`).
@interface RCImageOps : NSObject

/// Letterbox onto an exact HxW canvas (static ONNX export, CoreML EP).
/// Mirrors `yolo_onnx_runner.letterbox_exact`.
+ (RCLetterboxResult *)letterboxExact:(CVPixelBufferRef)pixelBuffer
                              targetW:(int)targetW
                              targetH:(int)targetH;

/// Stride-32 rect letterbox for the dynamic ONNX export (CPU parity path).
/// Mirrors `yolo_onnx_runner.letterbox`.
+ (RCLetterboxResult *)letterboxDynamic:(CVPixelBufferRef)pixelBuffer
                                  imgsz:(int)imgsz;

@end

NS_ASSUME_NONNULL_END
