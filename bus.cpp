#include "bus.hpp"
#include <cmath>

static std::queue<cv::Mat> recorded_frames;

void record_video_frames(cv::Mat& frame, const std::vector<cv::Point2f>& src_points){
    if(src_points.size() != 4)
        return;
    cv::Mat rectified = warp_perspective_rectified(frame, src_points);
    recorded_frames.push(frame);
}

cv::Mat warp_perspective_rectified(const cv::Mat& image, const std::vector<cv::Point2f>& src_points) {
    if (src_points.size() != 4) 
        return cv::Mat(); // 빈 결과 반환

    // 너비 계산
    float widthTop = cv::norm(src_points[0] - src_points[1]);
    float widthBottom = cv::norm(src_points[3] - src_points[2]);
    int target_width = (int)(widthTop > widthBottom ? widthTop : widthBottom);

    // 높이 계산
    float heightLeft = cv::norm(src_points[0] - src_points[3]);
    float heightRight = cv::norm(src_points[1] - src_points[2]);
    int target_height = (int)(heightLeft > heightRight ? heightLeft : heightRight);

    // 목표 정사각형 좌표 정의
    std::vector<cv::Point2f> target_form = {
        cv::Point2f(0, 0),
        cv::Point2f(target_width - 1, 0),
        cv::Point2f(target_width - 1, target_height - 1),
        cv::Point2f(0, target_height - 1)
    };

    // 투시변환 행렬 생성
    cv::Mat matrix = cv::getPerspectiveTransform(src_points, target_form);

    // 투시변환 적용
    cv::Mat rectified;
    cv::warpPerspective(image, rectified, matrix, cv::Size(target_width, target_height));

    return rectified;
}

//도로 색상을 흰색을 검은색으로 나머지는 흰색으로
cv::Mat mask_road_area(const cv::Mat& frame, const cv::Scalar& lower_bound, const cv::Scalar& upper_bound) {
    cv::Mat hsvFrame, mask, mask_inv, masked, output;

    // HSV 변환
    cv::cvtColor(frame, hsvFrame, cv::COLOR_BGR2HSV);

    // 범위 마스크
    cv::inRange(hsvFrame, lower_bound, upper_bound, mask);

    // 역마스크 생성 (마스크 외 영역: 도로가 아닌 부분 = 흰색 처리 대상)
    cv::bitwise_not(mask, mask_inv);

    // 출력 초기화 및 마스크 외 영역 흰색 처리
    masked = cv::Mat::zeros(frame.size(), CV_8UC3);
    masked.setTo(cv::Scalar(255, 255, 255), mask_inv);

    // 그레이 변환
    cv::bilateralFilter(masked, output, -1, 10, 5);
    cv::cvtColor(output, output, cv::COLOR_BGR2GRAY);

    return output;
}
