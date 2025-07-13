#include <iostream>
#include <opencv2/opencv.hpp>

#include "bus.hpp"

std::vector<cv::Point2f> clickedPoints;
void onMouse(int event, int x, int y, int flags, void* userdata);

int main(int argc, char** argv) {
  // 동영상 파일 경로 (인자로 받을 수도 있음)
  std::string videoPath = "test.MP4";

  cv::VideoCapture cap(videoPath);
  if (!cap.isOpened()) {
    std::cerr << "Error: Cannot open video file: " << videoPath << std::endl;
    return -1;
  }

  // 창 생성
  const std::string windowName = "Bus station";
  cv::namedWindow(windowName, cv::WINDOW_NORMAL);
  cv::resizeWindow(windowName, 600, 400);

  cv::setMouseCallback(windowName, onMouse, nullptr);

  cv::Mat frame, output;

  while (true) {
    cap >> frame;

    if (frame.empty()) break;

    // 도로 색상 추출 조건 (예: 채도 낮고 명도 낮은 영역)
    // S 채도: 0~60, V 명도: 0~100 정도로 설정
    cv::Scalar lower_bound(0, 0, 0);       // HSV lower bound
    cv::Scalar upper_bound(180, 60, 100);  // HSV upper bound

    if (clickedPoints.size() == 4) {
      cv::Mat warp_area = warp_perspective_rectified(frame, clickedPoints);
      output = mask_road_area(warp_area, lower_bound, upper_bound);
    } else {
      output = frame;
    }

    cv::imshow(windowName, output);

    if (cv::waitKey(30) >= 0) break;
  }

  // 자원 해제
  cap.release();
  cv::destroyAllWindows();

  return 0;
}

void onMouse(int event, int x, int y, int flags, void* userdata) {
  if (event == cv::EVENT_LBUTTONDOWN) {
    cv::Point2f point(static_cast<float>(x), static_cast<float>(y));
    
    if(clickedPoints.size() >= 4){
        clickedPoints.clear();
    }else{
        clickedPoints.push_back(point);
    }

    std::cout << "Clicked at: (" << point.x << ", " << point.y << ")"
              << clickedPoints.size() << std::endl;
  }
}
