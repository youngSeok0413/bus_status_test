#include <iostream>
#include <opencv2/opencv.hpp>
#include "bus.hpp"

std::vector<cv::Point2f> clickedPoints;
void onMouse(int event, int x, int y, int flags, void *userdata);

int main(int argc, char **argv)
{
  // 동영상 파일 경로 (인자로 받을 수도 있음)
  std::string videoPath = "test.MP4";

  cv::VideoCapture cap(videoPath);
  if (!cap.isOpened())
  {
    std::cerr << "Error: Cannot open video file: " << videoPath << std::endl;
    return -1;
  }

  // 창 생성
  const std::string windowName = "Bus station";
  cv::namedWindow(windowName, cv::WINDOW_NORMAL);
  cv::resizeWindow(windowName, 600, 400);

  cv::setMouseCallback(windowName, onMouse, nullptr);

  cv::Mat frame, output;

  cv::Scalar abs_lower_bound(0, 0, 0);      // absolute value
  cv::Scalar abs_upper_bound(180, 60, 100); // absolute value
  cv::Scalar rel_lower_bound;               // absolute value
  cv::Scalar rel_upper_bound;               // absolute value

  while (true)
  {
    cap >> frame;

    if (frame.empty())
      break;

    if (clickedPoints.size() == 4)
    {
      cv::Mat warp_area = warp_perspective_rectified(frame, clickedPoints);

      set_mask_of_road_area(warp_area, abs_lower_bound, abs_upper_bound, rel_lower_bound, rel_upper_bound);

      cv::Mat fgMask, blueMask, greenMask, redMask1, redMask2;
      cv::inRange(warp_area, rel_lower_bound, rel_upper_bound, fgMask);
      cv::bitwise_not(fgMask, fgMask);
      cv::inRange(warp_area, cv::Scalar(100,80,60), cv::Scalar(130,255,255), blueMask);
      cv::inRange(warp_area, cv::Scalar(40,70,60),  cv::Scalar(85,255,255),  greenMask);
      cv::inRange(warp_area, cv::Scalar(0,70,60),   cv::Scalar(10,255,255),  redMask1);
      cv::inRange(warp_area, cv::Scalar(170,70,60), cv::Scalar(180,255,255), redMask2);
      cv::Mat colorMask = (blueMask | greenMask | redMask1 | redMask2) & fgMask;
      
      cv::bitwise_and(warp_area, output, colorMask);
    }
    else
    {
      output = frame;
    }

    cv::imshow(windowName, output);

    if (cv::waitKey(30) >= 0)
      break;
  }

  // 자원 해제
  cap.release();
  cv::destroyAllWindows();

  return 0;
}

void onMouse(int event, int x, int y, int flags, void *userdata)
{
  if (event == cv::EVENT_LBUTTONDOWN)
  {
    cv::Point2f point(static_cast<float>(x), static_cast<float>(y));

    if (clickedPoints.size() >= 4)
    {
      clickedPoints.clear();
    }
    else
    {
      clickedPoints.push_back(point);
    }

    std::cout << "Clicked at: (" << point.x << ", " << point.y << ")"
              << clickedPoints.size() << std::endl;
  }
}
