#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <numeric>
#include <ctime>

cv::Mat frame; // 전역 변수로 현재 프레임 저장

bool isAchromatic(const cv::Vec3b &bgra, double threshold)
{
  // BGR → 정규화된 R, G, B (0~1 범위)
  double B = bgra[0];
  double G = bgra[1];
  double R = bgra[2];

  // 정규화 총합
  double sum = R + G + B;
  if (sum == 0)
    return true; // 완전 검정도 무채색

  // 정규화된 비율
  double r_ratio = R / sum;
  double g_ratio = G / sum;
  double b_ratio = B / sum;

  // 최대값과 최소값의 차이
  double max_val = std::max({r_ratio, g_ratio, b_ratio});
  double min_val = std::min({r_ratio, g_ratio, b_ratio});

  // RGB 비율이 거의 같으면 무채색
  return (max_val - min_val) < threshold;
}

double getAchromatic(const cv::Vec3b &bgra)
{
  // BGR → 정규화된 R, G, B (0~1 범위)
  double B = bgra[0];
  double G = bgra[1];
  double R = bgra[2];

  // 정규화 총합
  double sum = R + G + B;
  if (sum == 0)
    return true; // 완전 검정도 무채색

  // 정규화된 비율
  double r_ratio = R / sum;
  double g_ratio = G / sum;
  double b_ratio = B / sum;

  // 최대값과 최소값의 차이
  double max_val = std::max({r_ratio, g_ratio, b_ratio});
  double min_val = std::min({r_ratio, g_ratio, b_ratio});

  return max_val - min_val;
}

void getChromaticMask(const cv::Mat &bgrImage, cv::Mat &dst_mask, double threshold_1 = 0.15, int threshold_2 = 650)
{
  // 결과 마스크: 1채널 8비트 (0 = 무채색, 255 = 유채색)
  dst_mask = cv::Mat::zeros(bgrImage.size(), CV_8UC1);

  for (int y = 0; y < bgrImage.rows; ++y)
  {
    for (int x = 0; x < bgrImage.cols; ++x)
    {
      cv::Vec3b bgra = bgrImage.at<cv::Vec3b>(y, x);
      if (!isAchromatic(bgra, threshold_1) || bgra[0] + bgra[1] + bgra[2] > threshold_2)
      {
        // 유채색 → 마스크에 255
        dst_mask.at<uchar>(y, x) = 255;
      }
    }
  }
}

void getUnifiedMaskDynamicWhite(
    const cv::Mat &bgrImage,
    cv::Mat &dst_mask,
    double chroma_threshold = 0.15,
    double white_percentile = 90.0 // 상위 10% 밝기 이상이면 흰색
) //만약 local 하게 처리하면?
{
  CV_Assert(bgrImage.type() == CV_8UC3); // BGR 형식

  dst_mask = cv::Mat::zeros(bgrImage.size(), CV_8UC1);

  std::vector<std::pair<cv::Point, int>> achromatic_points;
  std::vector<int> achromatic_brightness;

  // 1차 분류 및 밝기 수집
  for (int y = 0; y < bgrImage.rows; ++y)
  {
    for (int x = 0; x < bgrImage.cols; ++x)
    {
      cv::Vec3b bgr = bgrImage.at<cv::Vec3b>(y, x);
      int brightness = bgr[0] + bgr[1] + bgr[2];

      if (!isAchromatic(bgr, chroma_threshold))
      {
        dst_mask.at<uchar>(y, x) = 255; // 유채색
      }
      else
      {
        achromatic_points.emplace_back(cv::Point(x, y), brightness);
        achromatic_brightness.push_back(brightness); //-> 함수 자체에 대해서 sort 함수를 만들것
      }
    }
  }

  // 무채색 밝기 기준 계산 (white_percentile 분위)
  if (!achromatic_brightness.empty())
  {
    std::unique(achromatic_brightness.begin(), achromatic_brightness.end());
    std::cout << achromatic_brightness.size() << std::endl;
    std::sort(achromatic_brightness.begin(), achromatic_brightness.end());
    int idx = std::clamp(
        static_cast<int>((white_percentile / 100.0) * achromatic_brightness.size()),
        0,
        static_cast<int>(achromatic_brightness.size()) - 1);

    int dynamic_brightness_thresh = achromatic_brightness[idx];

    // 밝은 무채색 → 흰색 간주
    for (auto &[pt, brightness] : achromatic_points)
    {
      if (brightness >= dynamic_brightness_thresh)
        dst_mask.at<uchar>(pt) = 255;
    }
  }
}

// 클릭 시 RGB 비율 계산 및 출력
void onMouse(int event, int x, int y, int, void *)
{
  if (event == cv::EVENT_LBUTTONDOWN)
  {
    if (x >= 0 && y >= 0 && x < frame.cols && y < frame.rows)
    {
      cv::Vec3b bgr = frame.at<cv::Vec3b>(y, x);
      int B = bgr[0], G = bgr[1], R = bgr[2];

      // 최소값 기준으로 정규화된 비율 계산
      double r = R;
      double g = G;
      double b = B;

      double sum = r + g + b;
      if (sum > 0)
      {
        double R_ratio = r / sum;
        double G_ratio = g / sum;
        double B_ratio = b / sum;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "픽셀 (" << x << ", " << y << ") → R:G:B = "
                  << R_ratio << " : "
                  << G_ratio << " : "
                  << B_ratio << std::endl;
      }
      else
      {
        std::cout << "픽셀 (" << x << ", " << y << ") → 완전한 검정색 (0, 0, 0)" << std::endl;
      }
    }

    std::vector<double> var;
    std::vector<int> lightnes;

    for (int y = 0; y < frame.rows; y++)
    {
      for (int x = 0; x < frame.cols; x++)
      {
        cv::Vec3b bgra = frame.at<cv::Vec3b>(y, x);
        var.push_back(getAchromatic(bgra));
        lightnes.push_back(bgra[0] + bgra[1] + bgra[2]);
      }
    }

    std::sort(var.begin(), var.end());
    std::sort(lightnes.begin(), lightnes.end());

    std::cout << "var : ";
    for (int i = 0; i < 20; i++)
    {
      std::cout << var[i] << ' ';
    }
    std::cout << std::accumulate(var.begin(), var.end(), 0.0) / var.size() << ' ';
    std::cout << var[var.size() / 2] << ' ';
    std::cout << std::endl;

    std::cout << "light : ";
    for (int i = 0; i < 20; i++)
    {
      std::cout << lightnes[i] << ' ';
    }
    std::cout << std::accumulate(lightnes.begin(), lightnes.end(), 0.0) / lightnes.size() << ' ';
    std::cout << lightnes[lightnes.size() / 2] << ' ';
    std::cout << std::endl;
  }
}

int main()
{
  // 트랙바 초기값
  int thresh_1 = 5;
  int thresh_2 = 90;

  // 영상 열기
  cv::VideoCapture cap("test.MP4"); // 웹캠 또는 "video.mp4"
  if (!cap.isOpened())
  {
    std::cerr << "❌ 비디오를 열 수 없습니다!" << std::endl;
    return -1;
  }

  // 트랙바 설정
  cv::namedWindow("Controls", cv::WINDOW_AUTOSIZE);
  cv::createTrackbar("th1", "Controls", &thresh_1, 100);
  cv::createTrackbar("th2", "Controls", &thresh_2, 100);
  cv::namedWindow("Result", cv::WINDOW_NORMAL);
  cv::resizeWindow("Result", 800, 600);
  cv::setMouseCallback("Result", onMouse); // 마우스 콜백 등록
  cv::namedWindow("Mask", cv::WINDOW_NORMAL);
  cv::resizeWindow("Mask", 800, 600);

  cv::Mat mask, lab, result;

  time_t start, end;
  while (true)
  {
    start = time(NULL);
    cap >> frame;
    if (frame.empty())
      break;

    
    cv::cvtColor(frame, lab, cv::COLOR_BGR2Lab);

    // 채널 분리
    std::vector<cv::Mat> lab_channels;
    cv::split(lab, lab_channels);

    // CLAHE 적용
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(2.0); // 대비 제한
    clahe->setTilesGridSize(cv::Size(8, 8));
    clahe->apply(lab_channels[0], lab_channels[0]); // L* 채널 //97% 85% 여전히 다름

    // 다시 합치고 변환
    cv::merge(lab_channels, lab);

    cv::Mat result;
    cv::cvtColor(lab, result, cv::COLOR_Lab2BGR);

    getUnifiedMaskDynamicWhite(frame, mask, (double)thresh_1 / 100, thresh_2); // before change -> color filter
    end = time(NULL);
    std::cout << (double)(end-start) << std::endl;
    // 결과 출력
    cv::imshow("Result", result);
    cv::imshow("Mask", mask);

    // 종료 키
    char key = (char)cv::waitKey(30);
    if (key == 27 || key == 'q')
      break;
  }

  cap.release();
  cv::destroyAllWindows();
  return 0;
}

// getChromaticMask(frame, mask, (double)thresh_1/100, thresh_2); // before change -> color filter
