#import "RCCourtDetector.h"
#import "RCCV.hpp"
#import <vector>
#import <cmath>
#import <limits>

using Line = cv::Vec4i;   // x1, y1, x2, y2

// ------------------------------------------------------------------ result ---

@implementation RCCourtResult
- (instancetype)initWithMask:(NSData *)mask width:(int)width height:(int)height success:(BOOL)success {
    if ((self = [super init])) { _mask = mask; _width = width; _height = height; _success = success; }
    return self;
}
@end

// ------------------------------------------------------- C++ line geometry ---
// Direct ports of the CourtDetector helper methods.

namespace {

const int MIN_BASELINE_LEN = 500;
const float BASELINE_WIDTH_RATIO = 0.6f;

struct LineEq { bool vertical; double slope; double intercept; }; // intercept = x1 when vertical

double polarAngleDeg(const Line &l) {
    double a = std::atan2((double)(l[3] - l[1]), (double)(l[2] - l[0])) * 180.0 / M_PI;
    if (a < 0) a += 360.0;
    return a;
}

LineEq lineEquation(const Line &l) {
    if (l[2] - l[0] == 0) return {true, 0.0, (double)l[0]};
    double slope = (double)(l[3] - l[1]) / (double)(l[2] - l[0]);
    return {false, slope, (double)l[1] - slope * (double)l[0]};
}

bool xInDomain(const Line &l, double x) {
    return std::min(l[0], l[2]) <= x && x <= std::max(l[0], l[2]);
}

// Returns NaN when the line is vertical (matches Python's None sentinel use).
double yAtX(const Line &l, double x) {
    LineEq e = lineEquation(l);
    if (e.vertical) return std::numeric_limits<double>::quiet_NaN();
    return e.slope * x + e.intercept;
}

// detect_court_lines
void detectCourtLines(const cv::Mat &frame,
                      std::vector<Line> &horizontal, std::vector<Line> &vertical,
                      std::vector<Line> &rightDiag, std::vector<Line> &leftDiag) {
    cv::Mat gray, blurred, canny, lab, white, dilated, refined;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(7, 7), 0);
    cv::Canny(blurred, canny, 50, 150);
    cv::cvtColor(frame, lab, cv::COLOR_BGR2LAB);
    cv::inRange(lab, cv::Scalar(145, 105, 105), cv::Scalar(255, 150, 150), white);
    cv::Mat k4 = cv::Mat::ones(4, 4, CV_8U);
    cv::dilate(canny, dilated, k4);
    cv::bitwise_and(white, dilated, refined);
    cv::morphologyEx(refined, refined, cv::MORPH_CLOSE, k4);

    const int h = frame.rows, w = frame.cols;
    cv::Mat roi = cv::Mat::zeros(h, w, CV_8U);
    int topCutoff = (int)(h * 0.35);
    std::vector<cv::Point> poly = {{0, h}, {0, topCutoff}, {w, topCutoff}, {w, h}};
    cv::fillPoly(roi, std::vector<std::vector<cv::Point>>{poly}, cv::Scalar(255));
    cv::Mat masked;
    cv::bitwise_and(refined, refined, masked, roi);

    std::vector<Line> lines;
    cv::HoughLinesP(masked, lines, 1, CV_PI / 180, 100, 100, 40);

    const double centerX = frame.cols / 2.0;
    for (const auto &l : lines) {
        int normalized = ((int)polarAngleDeg(l)) % 180;
        double midX = (l[0] + l[2]) / 2.0;
        bool right = midX >= centerX;
        if (normalized < 15 || normalized > 165) horizontal.push_back(l);
        else if (normalized > 75 && normalized < 105) vertical.push_back(l);
        else if (normalized >= 15 && normalized <= 75) { if (right) rightDiag.push_back(l); }
        else if (normalized >= 105 && normalized <= 165) { if (!right) leftDiag.push_back(l); }
    }
}

// merge_lines
std::vector<Line> mergeLines(const std::vector<Line> &lines, cv::Size shape,
                             cv::Size kernelSize, int iterations = 2, int minArea = 50) {
    std::vector<Line> out;
    if (lines.empty()) return out;
    cv::Mat mask = cv::Mat::zeros(shape.height, shape.width, CV_8U);
    for (const auto &l : lines) cv::line(mask, {l[0], l[1]}, {l[2], l[3]}, cv::Scalar(255), 3);
    cv::Mat kernel = cv::Mat::ones(kernelSize.height, kernelSize.width, CV_8U);
    cv::Mat closed;
    cv::morphologyEx(mask, closed, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), iterations);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(closed, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (auto &c : contours) {
        if (cv::contourArea(c) < minArea) continue;
        std::vector<cv::Point> hull;
        cv::convexHull(c, hull);
        double maxDist = 0; cv::Point p1, p2; bool found = false;
        for (auto &a : hull) for (auto &b : hull) {
            double d = cv::norm(a - b);
            if (d > maxDist) { maxDist = d; p1 = a; p2 = b; found = true; }
        }
        if (found) out.push_back(Line(p1.x, p1.y, p2.x, p2.y));
    }
    return out;
}

// find_baseline
bool findBaseline(const std::vector<Line> &horizontal, Line &out) {
    std::vector<Line> candidates;
    for (const auto &l : horizontal) if (std::abs(l[2] - l[0]) >= MIN_BASELINE_LEN) candidates.push_back(l);
    if (candidates.empty()) return false;
    int maxWidth = 0;
    for (const auto &l : candidates) maxWidth = std::max(maxWidth, std::abs(l[2] - l[0]));
    double bestMeanY = -1; bool found = false;
    for (const auto &l : candidates) {
        if (std::abs(l[2] - l[0]) < BASELINE_WIDTH_RATIO * maxWidth) continue;
        double meanY = (l[1] + l[3]) / 2.0;
        if (!found || meanY > bestMeanY) { bestMeanY = meanY; out = l; found = true; }
    }
    return found;
}

bool validateSideline(const Line &cand, const Line &baseline, int imageWidth) {
    double baselineWidth = std::abs(baseline[2] - baseline[0]);
    double pct = (baselineWidth / imageWidth) * 100.0;
    double tolerance = (pct <= 98.5) ? 100 : 150;
    double candBottomY = std::max(cand[1], cand[3]);
    double baselineY = (baseline[1] + baseline[3]) / 2.0;
    return std::abs(candBottomY - baselineY) <= tolerance;
}

// _find_outer_line
bool findOuterLine(const std::vector<Line> &lines, Line &out) {
    if (lines.size() < 2) { if (!lines.empty()) { out = lines[0]; return true; } return false; }
    const Line &l1 = lines[0], &l2 = lines[1];
    double midX1 = (l1[0] + l1[2]) / 2.0;
    if (xInDomain(l2, midX1)) {
        double y1 = yAtX(l1, midX1), y2 = yAtX(l2, midX1);
        if (!std::isnan(y1) && !std::isnan(y2)) { out = (y1 < y2) ? l1 : l2; return true; }
    }
    double midX2 = (l2[0] + l2[2]) / 2.0;
    if (xInDomain(l1, midX2)) {
        double y1 = yAtX(l1, midX2), y2 = yAtX(l2, midX2);
        if (!std::isnan(y1) && !std::isnan(y2)) { out = (y1 < y2) ? l1 : l2; return true; }
    }
    double avg1 = (l1[1] + l1[3]) / 2.0, avg2 = (l2[1] + l2[3]) / 2.0;
    out = (avg1 < avg2) ? l1 : l2;
    return true;
}

// _process_full_width_baseline_case
bool fullWidthBaselineCase(const std::vector<Line> &diagonals, const Line &baseline,
                           const std::string &side, Line &out) {
    double baselineY = (baseline[1] + baseline[3]) / 2.0;
    const double tol = 100;
    std::vector<Line> close;
    for (const auto &l : diagonals) if (std::abs(std::max(l[1], l[3]) - baselineY) < tol) close.push_back(l);
    if (close.size() == 1) {
        const Line &singles = close[0];
        double midY = (singles[1] + singles[3]) / 2.0;
        LineEq se = lineEquation(singles);
        double xRef = se.vertical ? singles[0] : (midY - se.intercept) / se.slope;
        bool found = false; double minDist = std::numeric_limits<double>::infinity();
        for (const auto &l : diagonals) {
            bool isClose = false; for (const auto &c : close) if (c == l) { isClose = true; break; }
            if (isClose) continue;
            LineEq e = lineEquation(l);
            double xCand = e.vertical ? l[0] : (midY - e.intercept) / e.slope;
            bool outward = (side == "right") ? (xCand > xRef) : (xCand < xRef);
            if (outward) { double d = std::abs(xCand - xRef); if (d < minDist) { minDist = d; out = l; found = true; } }
        }
        return found;
    } else if (close.size() == 2) {
        const Line &l1 = close[0], &l2 = close[1];
        double midX = (l1[0] + l1[2]) / 2.0;
        if (!(std::min(l2[0], l2[2]) <= midX && midX <= std::max(l2[0], l2[2]))) midX = (l2[0] + l2[2]) / 2.0;
        LineEq e1 = lineEquation(l1), e2 = lineEquation(l2);
        double y1 = e1.vertical ? l1[1] : e1.slope * midX + e1.intercept;
        double y2 = e2.vertical ? l2[1] : e2.slope * midX + e2.intercept;
        out = (y1 < y2) ? l1 : l2;
        return true;
    }
    return false;
}

// _process_partial_baseline_case
bool partialBaselineCase(const std::vector<Line> &diagonals, const Line &baseline,
                         const std::string &side, Line &out) {
    double baselineY = (baseline[1] + baseline[3]) / 2.0;
    double baselineEndX = (side == "right") ? std::max(baseline[0], baseline[2]) : std::min(baseline[0], baseline[2]);
    double minDist = std::numeric_limits<double>::infinity(); bool found = false;
    for (const auto &l : diagonals) {
        double nearEndX = (l[1] > l[3]) ? l[0] : l[2];
        double nearEndY = std::max(l[1], l[3]);
        double d = std::sqrt((nearEndX - baselineEndX) * (nearEndX - baselineEndX)
                             + (nearEndY - baselineY) * (nearEndY - baselineY));
        if (d < minDist) { minDist = d; out = l; found = true; }
    }
    return found;
}

// process_side_decision_tree
bool processSide(const std::vector<Line> &diagonals, const Line &baseline,
                 int imageWidth, const std::string &side, Line &out) {
    size_t count = diagonals.size();
    if (count <= 1) return false;
    if (count == 2) {
        Line outer; if (!findOuterLine(diagonals, outer)) return false;
        if (validateSideline(outer, baseline, imageWidth)) { out = outer; return true; }
        return false;
    }
    double pct = (std::abs(baseline[2] - baseline[0]) / (double)imageWidth) * 100.0;
    if (pct > 98.5) {
        Line cand; if (fullWidthBaselineCase(diagonals, baseline, side, cand)) { out = cand; return true; }
        return false;
    }
    Line cand; if (!partialBaselineCase(diagonals, baseline, side, cand)) return false;
    if (validateSideline(cand, baseline, imageWidth)) { out = cand; return true; }
    return false;
}

// estimate_playable_court_area
cv::Mat estimateOutMask(const Line &left, const Line &right, const Line &baseline, cv::Size shape) {
    const double BASE_HORIZONTAL_SHIFT = 100;
    int w = shape.width, h = shape.height;
    double baselineWidth = std::abs(baseline[2] - baseline[0]);
    double dynamicShift = BASE_HORIZONTAL_SHIFT * (baselineWidth / w);

    LineEq le = lineEquation(left), re = lineEquation(right);
    double leftShiftInt = le.vertical ? le.intercept
        : le.intercept - dynamicShift / std::sqrt(1 + le.slope * le.slope);
    double rightShiftInt = re.vertical ? re.intercept
        : re.intercept - dynamicShift / std::sqrt(1 + re.slope * re.slope);

    cv::Mat mask = cv::Mat::zeros(h, w, CV_8U);
    const double posInf = std::numeric_limits<double>::infinity();
    const double negInf = -posInf;
    for (int x = 0; x < w; ++x) {
        double leftThr, rightThr;
        if (!le.vertical) leftThr = le.slope * x + leftShiftInt;
        else leftThr = (x < (left[0] - dynamicShift)) ? posInf : negInf;
        if (!re.vertical) rightThr = re.slope * x + rightShiftInt;
        else rightThr = (x > (right[0] + dynamicShift)) ? posInf : negInf;
        double thr = std::max(leftThr, rightThr);     // out if left OR right
        int yEnd = thr >= h ? h : (thr <= 0 ? 0 : (int)std::ceil(thr));
        for (int y = 0; y < yEnd; ++y) if (y < thr) mask.at<uchar>(y, x) = 255;
    }
    return mask;
}

// extract_clean_frame (homography inpaint of player regions using a reference frame)
cv::Mat cleanFrame(const cv::Mat &base, const cv::Mat *reference, const std::vector<cv::Rect> &baseBoxes) {
    if (reference == nullptr || reference->empty()) return base;
    cv::Ptr<cv::ORB> orb = cv::ORB::create(1000);
    std::vector<cv::KeyPoint> kp1, kp2; cv::Mat des1, des2;
    orb->detectAndCompute(base, cv::noArray(), kp1, des1);
    orb->detectAndCompute(*reference, cv::noArray(), kp2, des2);
    if (des1.empty() || des2.empty()) return base;

    cv::BFMatcher bf(cv::NORM_HAMMING, true);
    std::vector<cv::DMatch> matches;
    bf.match(des1, des2, matches);
    std::sort(matches.begin(), matches.end(), [](const cv::DMatch &a, const cv::DMatch &b){ return a.distance < b.distance; });
    size_t keep = std::min<size_t>(100, matches.size());
    if (keep < 10) return base;

    std::vector<cv::Point2f> srcPts, dstPts;
    for (size_t i = 0; i < keep; ++i) {
        srcPts.push_back(kp1[matches[i].queryIdx].pt);
        dstPts.push_back(kp2[matches[i].trainIdx].pt);
    }
    cv::Mat M = cv::findHomography(dstPts, srcPts, cv::RANSAC, 5.0);
    if (M.empty()) return base;

    cv::Mat warped;
    cv::warpPerspective(*reference, warped, M, base.size());
    cv::Mat occ = cv::Mat::zeros(base.rows, base.cols, CV_8U);
    for (const auto &b : baseBoxes) cv::rectangle(occ, b, cv::Scalar(255), cv::FILLED);
    cv::Mat k5 = cv::Mat::ones(5, 5, CV_8U);
    cv::dilate(occ, occ, k5);
    cv::Mat clean = base.clone();
    warped.copyTo(clean, occ);
    return clean;
}

} // namespace

// ---------------------------------------------------------------- Obj-C API ---

@implementation RCCourtDetector

+ (RCCourtResult *)detectWithBaseFrame:(CVPixelBufferRef)baseFrame
                             baseBoxes:(NSArray<NSValue *> *)baseBoxes
                        referenceFrame:(CVPixelBufferRef)referenceFrame {
    cv::Mat base = rc::bgrFromPixelBuffer(baseFrame);
    const int w = base.cols, h = base.rows;

    std::vector<cv::Rect> boxes;
    for (NSValue *v in baseBoxes) {
        CGRect r = v.CGRectValue;
        boxes.emplace_back((int)r.origin.x, (int)r.origin.y, (int)r.size.width, (int)r.size.height);
    }

    cv::Mat ref;
    if (referenceFrame != NULL) ref = rc::bgrFromPixelBuffer(referenceFrame);
    cv::Mat clean = cleanFrame(base, ref.empty() ? nullptr : &ref, boxes);

    std::vector<Line> horizontal, vertical, rightDiag, leftDiag;
    detectCourtLines(clean, horizontal, vertical, rightDiag, leftDiag);
    auto mHoriz = mergeLines(horizontal, clean.size(), cv::Size(30, 5));   // (cols,rows) = np (5,30)
    auto mRight = mergeLines(rightDiag, clean.size(), cv::Size(2, 2));
    auto mLeft = mergeLines(leftDiag, clean.size(), cv::Size(2, 2));

    Line baseline;
    if (!findBaseline(mHoriz, baseline)) {
        return [self failureForWidth:w height:h];
    }
    Line rightSide, leftSide;
    bool rightOK = processSide(mRight, baseline, clean.cols, "right", rightSide);
    bool leftOK = processSide(mLeft, baseline, clean.cols, "left", leftSide);
    if (!rightOK || !leftOK) return [self failureForWidth:w height:h];

    cv::Mat mask = estimateOutMask(leftSide, rightSide, baseline, clean.size());
    bool any = cv::countNonZero(mask) > 0;
    NSData *data = [NSData dataWithBytes:mask.data length:(NSUInteger)(mask.total() * mask.elemSize())];
    return [[RCCourtResult alloc] initWithMask:data width:w height:h success:any];
}

+ (RCCourtResult *)failureForWidth:(int)w height:(int)h {
    NSData *zeros = [NSMutableData dataWithLength:(NSUInteger)w * h];
    return [[RCCourtResult alloc] initWithMask:zeros width:w height:h success:NO];
}

+ (RCCourtResult *)defaultMaskFromPNGPath:(NSString *)pngPath width:(int)width height:(int)height {
    cv::Mat base = cv::imread(pngPath.UTF8String, cv::IMREAD_GRAYSCALE);
    if (base.empty()) return [self failureForWidth:width height:height];
    if (base.cols != width || base.rows != height) {
        cv::resize(base, base, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
    }
    cv::Mat mask;
    cv::threshold(base, mask, 127, 255, cv::THRESH_BINARY);
    NSData *data = [NSData dataWithBytes:mask.data length:(NSUInteger)(mask.total() * mask.elemSize())];
    return [[RCCourtResult alloc] initWithMask:data width:width height:height success:YES];
}

@end
