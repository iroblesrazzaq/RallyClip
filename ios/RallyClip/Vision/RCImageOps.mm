#import "RCImageOps.h"
#import "RCCV.hpp"

@implementation RCLetterboxResult
- (instancetype)initWithTensor:(NSData *)tensor width:(int)width height:(int)height
                         ratio:(float)ratio padLeft:(int)padLeft padTop:(int)padTop
                     origWidth:(int)origWidth origHeight:(int)origHeight {
    if ((self = [super init])) {
        _tensor = tensor; _width = width; _height = height; _ratio = ratio;
        _padLeft = padLeft; _padTop = padTop; _origWidth = origWidth; _origHeight = origHeight;
    }
    return self;
}
@end

@implementation RCImageOps

+ (RCLetterboxResult *)letterbox:(CVPixelBufferRef)pixelBuffer
                         targetW:(int)targetW targetH:(int)targetH
                           exact:(BOOL)exact imgsz:(int)imgsz {
    cv::Mat bgr = rc::bgrFromPixelBuffer(pixelBuffer);
    const int origW = bgr.cols, origH = bgr.rows;
    rc::LetterboxGeom geom{};
    cv::Mat lb = rc::letterbox(bgr, targetW, targetH, exact ? true : false, 32, geom);
    const int H = lb.rows, W = lb.cols;
    NSMutableData *data = [NSMutableData dataWithLength:(NSUInteger)(3 * H * W) * sizeof(float)];
    rc::packTensorRGB(lb, (float *)data.mutableBytes);
    return [[RCLetterboxResult alloc] initWithTensor:data width:W height:H
                                               ratio:geom.ratio padLeft:geom.padLeft padTop:geom.padTop
                                           origWidth:origW origHeight:origH];
}

+ (RCLetterboxResult *)letterboxExact:(CVPixelBufferRef)pixelBuffer
                              targetW:(int)targetW targetH:(int)targetH {
    return [self letterbox:pixelBuffer targetW:targetW targetH:targetH exact:YES imgsz:0];
}

+ (RCLetterboxResult *)letterboxDynamic:(CVPixelBufferRef)pixelBuffer imgsz:(int)imgsz {
    return [self letterbox:pixelBuffer targetW:imgsz targetH:imgsz exact:NO imgsz:imgsz];
}

@end
