#ifndef BUS_HPP
#define BUS_HPP

#include <vector>
#include <opencv2/opencv.hpp>

/**
Bus station status algorithm
flow의 유무로 bus algorithm 추측
flow 영역을 삼등분으로 할 것
 */

#define BUS_PLATFORM_ALREADY_EMPTY 0
#define BUS_PLATFORM_ALREADY_FILLED 1
#define BUS_PLATFORM_ABOUT_EMPTY 0
#define BUS_PLATFORM_ABOUT_FILLED 1 

void record_video_frames(cv::Mat& frame);
cv::Mat warp_perspective_rectified(const cv::Mat& image, const std::vector<cv::Point2f>& src_points);
cv::Mat mask_road_area(const cv::Mat& frame, const cv::Scalar& lower_bound, const cv::Scalar& upper_bound);

#endif