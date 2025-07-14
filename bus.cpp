#include "bus.hpp"
#include <cmath>

static std::queue<cv::Mat> recorded_frames;
static std::vector<cv::Mat> frame_per_platform;

/*static cv::Scalar road_mask_lower_boundary(0, 0, 0);       // HSV lower bound
static cv::Scalar road_mask_upper_boundary(180, 60, 100);  // HSV upper bound
*/

void record_video_frames(cv::Mat &frame)
{
    recorded_frames.push(frame);
}

cv::Mat warp_perspective_rectified(const cv::Mat &image, const std::vector<cv::Point2f> &src_points)
{
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
        cv::Point2f(0, target_height - 1)};

    // 투시변환 행렬 생성
    cv::Mat matrix = cv::getPerspectiveTransform(src_points, target_form);

    // 투시변환 적용
    cv::Mat rectified;
    cv::warpPerspective(image, rectified, matrix, cv::Size(target_width, target_height));

    return rectified;
}

// 도로 색상을 흰색을 검은색으로 나머지는 흰색으로
cv::Mat mask_road_area(const cv::Mat &frame, const cv::Scalar &lower_bound, const cv::Scalar &upper_bound)
{
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

cv::Mat mask_road_area_adaptive(const cv::Mat &frame)
{
    cv::Mat gray, blurred, adaptive_mask, mask_inv, masked, output;

    // 1. Grayscale 변환
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // 2. 잡음 제거 (Gaussian Blur)
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

    // 3. Adaptive Threshold (도로 어두운 영역 검출용)
    // 도로는 상대적으로 어두우므로 THRESH_BINARY_INV 사용
    cv::adaptiveThreshold(blurred, adaptive_mask,
                          255,
                          cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv::THRESH_BINARY_INV,
                          15, 10); // blockSize=15, C=10 은 경험적 값

    // 4. 역마스크: 도로가 아닌 영역 (밝은 영역) → 흰색 처리 대상
    cv::bitwise_not(adaptive_mask, mask_inv);

    // 5. 마스킹 적용 (비도로 영역 흰색 처리)
    masked = cv::Mat::zeros(frame.size(), CV_8UC3);
    masked.setTo(cv::Scalar(255, 255, 255), mask_inv);

    // 6. Bilateral 필터로 부드럽게
    cv::bilateralFilter(masked, output, -1, 10, 5);

    // 7. Grayscale 변환 후 반환
    cv::cvtColor(output, output, cv::COLOR_BGR2GRAY);

    return output;
}

cv::Mat mask_road_area_lab(const cv::Mat &frame, 
    const cv::Scalar &lower_bound, const cv::Scalar &upper_bound)
{
    cv::Mat labFrame, mask, mask_inv, masked, output;

    // BGR → Lab 색공간 변환
    cv::cvtColor(frame, labFrame, cv::COLOR_BGR2Lab);

    // Lab 범위에 따라 마스크 생성 (도로 색상 조건에 해당하는 픽셀 = 흰색)
    cv::inRange(labFrame, lower_bound, upper_bound, mask);

    // 역마스크 생성: 도로가 아닌 부분 = 흰색 처리 대상
    cv::bitwise_not(mask, mask_inv);

    // 출력 초기화 후 마스크 외 영역 흰색 처리
    masked = cv::Mat::zeros(frame.size(), CV_8UC3);
    masked.setTo(cv::Scalar(255, 255, 255), mask_inv);

    // 필터링 → 부드럽게 처리
    cv::bilateralFilter(masked, output, -1, 10, 5);

    // 흑백 변환
    cv::cvtColor(output, output, cv::COLOR_BGR2GRAY);

    return output;
}

// frame must me hsv form
void set_mask_of_road_area(const cv::Mat &hsvFrame,
                           cv::Scalar &abs_lower_bound, cv::Scalar &abs_upper_bound,
                           cv::Scalar &lower_bound, cv::Scalar &upper_bound)
{
    int rows, columns, total_pixels, mean_saturation, mean_value;

    // get mean saturation and value
    rows = hsvFrame.rows;
    columns = hsvFrame.cols;
    total_pixels = rows * columns;
    for (int i = 0; i < rows; i++)
    {
        const cv::Vec3b *ptr = hsvFrame.ptr<cv::Vec3b>(i);
        for (int j = 0; j < columns; ++j)
        {
            mean_saturation += ptr[j][1]; // S
            mean_value += ptr[j][2];      // V
        }
    }
    mean_saturation = (int)(mean_saturation / total_pixels);
    mean_value = (int)(mean_value / total_pixels);

    // adapted value
    lower_bound = cv::Scalar(abs_lower_bound[0], 
        mean_saturation * abs_lower_bound[1]/128, 
        mean_value * abs_lower_bound[2]/128);
    std::cout << "L: " << lower_bound[1] << " " << lower_bound[2] << std::endl;
    upper_bound = cv::Scalar(abs_upper_bound[0], 
        mean_saturation * abs_upper_bound[1]/128, 
        mean_value * abs_upper_bound[2]/128);
    std::cout << "U: " << upper_bound[1] << " " << upper_bound[2] << std::endl;
}