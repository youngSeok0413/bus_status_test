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

void record_video_frames(cv::Mat &frame);
void divide_platform_section(int n);
cv::Mat warp_perspective_rectified(const cv::Mat &image,
                                   const std::vector<cv::Point2f> &src_points);
cv::Mat mask_road_area(const cv::Mat &frame,
                       const cv::Scalar &lower_bound, const cv::Scalar &upper_bound);
cv::Mat mask_road_area_adaptive(const cv::Mat &frame);
cv::Mat mask_road_area_lab(const cv::Mat &frame,
                           const cv::Scalar &lower_bound, const cv::Scalar &upper_bound);
void set_mask_of_road_area(const cv::Mat &hsvFrame,
                           cv::Scalar &abs_lower_bound, cv::Scalar &abs_upper_bound,
                           cv::Scalar &lower_bound, cv::Scalar &upper_bound); // frame must be hsv form

/**
 * H : 0 ~ 180
 * S : 0 ~ 255
 * V : 0 ~ 255
 */

#endif