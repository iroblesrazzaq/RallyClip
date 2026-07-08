#import <Foundation/Foundation.h>
#import <CoreVideo/CoreVideo.h>
#import <CoreGraphics/CoreGraphics.h>

NS_ASSUME_NONNULL_BEGIN

/// Binary "out" court mask (255 = outside the playable area) + detection status.
@interface RCCourtResult : NSObject
@property (nonatomic, readonly) NSData *mask;   // uint8, length width*height (row-major)
@property (nonatomic, readonly) int width;
@property (nonatomic, readonly) int height;
@property (nonatomic, readonly) BOOL success;   // NO if lines/baseline/sidelines not found
@end

/// Obj-C++/OpenCV port of `preprocessing/court_detector_impl.py`.
///
/// The Swift driver supplies the frames (AVFoundation) and player boxes (the
/// pose runner doubles as the person detector, class 0), then this class does
/// the OpenCV work: homography clean-frame → line detection → baseline/sideline
/// decision tree → "out" mask. cv2 calls map 1:1 to the Python.
@interface RCCourtDetector : NSObject

/// Detect the court "out" mask from one sampled timestamp.
/// - baseFrame: frame at the target time (32BGRA).
/// - baseBoxes: person boxes in `baseFrame`, as CGRect(x,y,w,h) in source pixels.
/// - referenceFrame: a nearby frame with the players elsewhere (or nil to skip
///   the homography clean-frame step and use `baseFrame` directly).
+ (RCCourtResult *)detectWithBaseFrame:(CVPixelBufferRef)baseFrame
                             baseBoxes:(NSArray<NSValue *> *)baseBoxes
                        referenceFrame:(nullable CVPixelBufferRef)referenceFrame;

/// The empirical fallback mask (bundled `default_court_mask.png`), resized to the
/// frame. Mirrors `data_preprocessor._load_default_court_mask`.
+ (RCCourtResult *)defaultMaskFromPNGPath:(NSString *)pngPath
                                    width:(int)width
                                   height:(int)height;

@end

NS_ASSUME_NONNULL_END
