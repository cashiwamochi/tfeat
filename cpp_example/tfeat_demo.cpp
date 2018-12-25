#include <torch/script.h> // One-stop header.
// #include <torch/extension.h>
#include <torch/torch.h>

#include <iostream>
#include <vector>
#include <memory>
#include <cmath>

#include <opencv2/opencv.hpp>

#define PI 3.14159265

std::pair<cv::Mat, cv::Mat> loadDemoImages
  (const std::pair<std::string, std::string>& pstr_image_names) 
  {
  cv::Mat img0 = cv::imread(pstr_image_names.first, 0);
  cv::Mat img1 = cv::imread(pstr_image_names.second, 0);

  return std::make_pair(img0, img1);
  }

std::vector<cv::Mat> extractPatches(
                      const cv::Mat& m_image,
                      const std::vector<cv::KeyPoint>& v_kpts,
                      const float f_mag_factor = 3.0) 
{
  /*
   * original python source code
    patches = []
        for kp in kpts:
            x,y = kp.pt
            s = kp.size
            a = kp.angle

            s = mag_factor * s / N
            cos = math.cos(a * math.pi / 180.0)
            sin = math.sin(a * math.pi / 180.0)

            M = np.matrix([
                [+s * cos, -s * sin, (-s * cos + s * sin) * N / 2.0 + x],
                [+s * sin, +s * cos, (-s * sin - s * cos) * N / 2.0 + y]])

            patch = cv2.warpAffine(img, M, (N, N),
                                 flags=cv2.WARP_INVERSE_MAP + \
                                 cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS)

            patches.append(patch)
            
        patches = torch.from_numpy(np.asarray(patches)).float()
        patches = torch.unsqueeze(patches,1)
  */

  std::vector<cv::Mat> vm_patches;
  vm_patches.reserve(v_kpts.size());
  unsigned int N = 32;

  for(auto kp : v_kpts) {
    const cv::Point2f pt = kp.pt;
    const float f_angle = kp.angle;
    const float f_scale = f_mag_factor * kp.size / float(N);
    const float f_cos = std::cos(f_angle * PI / 180.0);
    const float f_sin = std::sin(f_angle * PI / 180.0);

    const cv::Mat m_affine = (cv::Mat_<float>(2,3) 
                      << f_scale*f_cos, -f_scale*f_sin, f_scale*(-f_cos + f_sin)*float(N)/2.0 + pt.x,
                         f_scale*f_sin, f_scale*f_cos, f_scale*(-f_sin - f_cos)*float(N)/2.0 + pt.y);

    cv::Mat m_patch = cv::Mat::zeros(N, N, m_image.type());
    cv::warpAffine(m_image, m_patch, m_affine, m_patch.size(),
                   cv::WARP_INVERSE_MAP | cv::INTER_CUBIC | cv::WARP_FILL_OUTLIERS);
    vm_patches.push_back(m_patch);
  }
  
  return vm_patches;
}

std::pair< std::vector<cv::Mat>,std::vector<cv::Mat> > extractKeyPointPatches
  (std::pair<cv::Mat, cv::Mat>& pm_images, 
   std::vector<cv::KeyPoint>& v_kpts0, std::vector<cv::KeyPoint>& v_kpts1) 
{
  cv::Ptr<cv::BRISK> brisk = cv::BRISK::create();
  cv::Mat m_descriptors0, m_descriptors1; 
  std::vector<cv::KeyPoint> _v_kpts0, _v_kpts1;
  brisk->detectAndCompute(pm_images.first, cv::noArray(), _v_kpts0, m_descriptors0);
  brisk->detectAndCompute(pm_images.second, cv::noArray(), _v_kpts1, m_descriptors1);

  v_kpts0 = _v_kpts0;
  v_kpts1 = _v_kpts1;

  cv::BFMatcher* matcher = new cv::BFMatcher(cv::NORM_HAMMING, false);
  std::vector< std::vector<cv::DMatch> > vv_matches_2nn_01;
  matcher->knnMatch(m_descriptors0, m_descriptors1, vv_matches_2nn_01, 2);

  std::cout << m_descriptors0.rows << " x " << m_descriptors0.cols << std::endl;
  std::cout << m_descriptors1.rows << " x " << m_descriptors1.cols << std::endl;

  if (true) {
    std::vector<cv::DMatch> v_good_matches_01;
    for (size_t i = 0; i < vv_matches_2nn_01.size(); i++){
      if (vv_matches_2nn_01[i][0].distance < 0.8*vv_matches_2nn_01[i][1].distance) {
        v_good_matches_01.push_back(vv_matches_2nn_01[i][0]);
      }
    }
    cv::Mat m_output;
    cv::drawMatches(pm_images.first,v_kpts0,pm_images.second,v_kpts1,v_good_matches_01,m_output);
    cv::imshow("matches_by_BRISK", m_output);
    cv::waitKey(0);
  }

  std::pair< std::vector<cv::Mat>, std::vector<cv::Mat> >
    pvv_patches = std::make_pair(extractPatches(pm_images.first, v_kpts0), extractPatches(pm_images.second, v_kpts1));

  return pvv_patches;
}

int main(int argc, char* argv[]) {
  if (argc != 4){
    std::cout << "usage : this.out [/path/to/model.pt] [/path/to/image0] [/path/to/image1]";
    std::cout << std::endl;
    return -1;
  }

  // Loading your model
  std::string s_model_name = argv[1];
  std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(s_model_name);

  assert(module != nullptr);
  std::cout << ">>> Loaded your model " << s_model_name << std::endl;;

  // Loading your images
  std::string s_image_name0 = argv[2];
  std::string s_image_name1 = argv[3];

  std::cout << ">>> Loading " << s_image_name0 << " " << s_image_name1;
  std::pair<cv::Mat, cv::Mat> pm_images
    = loadDemoImages(std::make_pair(s_image_name0,s_image_name1));
  std::cout << " Done! " << std::endl;

#if 0
  cv::imshow("test0", pm_images.first);
  cv::imshow("test1", pm_images.second);
  cv::waitKey(0);
#endif

  std::cout << ">>> Extracintg ... " << std::endl;
  std::vector<cv::KeyPoint> v_kpts0, v_kpts1;
  std::pair< std::vector<cv::Mat>,std::vector<cv::Mat> > 
    pvv_patches = extractKeyPointPatches(pm_images, v_kpts0, v_kpts1);


  std::cout << "KP Num(0) = " << v_kpts0.size() << std::endl;
  std::cout << "KP Num(1) = " << v_kpts1.size() << std::endl;

  std::cout << ">>> Forward ... " << std::endl;

  std::vector<int64_t> dims{static_cast<int64_t>(v_kpts0.size()), 
                            static_cast<int64_t>(1), // 1
                            static_cast<int64_t>(32), // 32
                            static_cast<int64_t>(32)}; // 32

  std::vector<cv::Mat> vm_patches0 = pvv_patches.first;
  cv::Mat m_descriptors0 = cv::Mat::zeros(vm_patches0.size(), 128, CV_32FC1);
  std::cout << m_descriptors0.rows << ":" << m_descriptors0.cols << std::endl;

  std::vector<at::Tensor> vt_patches0;
  cv::Mat mf_patch_concat = cv::Mat::zeros(32*vm_patches0.size(), 32, CV_32FC1);
  vt_patches0.reserve(vm_patches0.size());
  for(int i=0; i < vm_patches0.size(); i++) {
    cv::Mat mf_patch;
    vm_patches0[i].convertTo(mf_patch, CV_32FC1);
    mf_patch.copyTo(mf_patch_concat.rowRange(i*32,(i+1)*32));
  }

  at::TensorOptions options(at::kFloat);
  at::Tensor tensor_patch = torch::from_blob(mf_patch_concat.data, at::IntList(dims), options);
  std::cout << "============" << std::endl;

  at::Tensor t_descriptor0 = module->forward({tensor_patch}).toTensor().clone();

  // std::cout << t_descriptor0.size(0) << std::endl;
  // std::cout << t_descriptor0.size(1) << std::endl;
  // std::cout << t_descriptor0.slice(0,0,5) << std::endl;
  cv::Mat m_feature0(cv::Size(128, vm_patches0.size()), CV_32FC1, t_descriptor0.data<float>());
  std::cout << m_feature0 << std::endl;

  m_descriptors0 = m_feature0.clone();

  std::vector<cv::Mat> vm_patches1 = pvv_patches.second;
  cv::Mat m_descriptors1 = cv::Mat::zeros(vm_patches1.size(), 128, CV_32FC1);

  std::vector<at::Tensor> vt_patches1;
  mf_patch_concat.release();
  mf_patch_concat = cv::Mat::zeros(32*vm_patches1.size(), 32, CV_32FC1);
  vt_patches1.reserve(vm_patches1.size());
  for(int i=0; i < vm_patches1.size(); i++) {
    cv::Mat mf_patch;
    vm_patches1[i].convertTo(mf_patch, CV_32FC1);
    mf_patch.copyTo(mf_patch_concat.rowRange(i*32,(i+1)*32));
  }

  dims.clear();
  dims = std::vector<int64_t>{static_cast<int64_t>(v_kpts1.size()), 
                              static_cast<int64_t>(1), // 1
                              static_cast<int64_t>(32), // 32
                              static_cast<int64_t>(32)}; // 32
  tensor_patch = torch::from_blob(mf_patch_concat.data, at::IntList(dims), options);
  std::cout << "============" << std::endl;

  at::Tensor t_descriptor1 = module->forward({tensor_patch}).toTensor().clone();

  // std::cout << t_descriptor0.size(0) << std::endl;
  // std::cout << t_descriptor0.size(1) << std::endl;
  // std::cout << t_descriptor0.slice(0,0,5) << std::endl;
  cv::Mat m_feature1(cv::Size(128, vm_patches1.size()), CV_32FC1, t_descriptor1.data<float>());
  std::cout << m_feature1 << std::endl;

  m_descriptors1 = m_feature1.clone();
  
  cv::BFMatcher* matcher = new cv::BFMatcher(cv::NORM_L2, false);
  std::vector< std::vector<cv::DMatch> > vv_matches_2nn_01;
  matcher->knnMatch(m_descriptors0, m_descriptors1, vv_matches_2nn_01, 2);
  if (true) {
    std::vector<cv::DMatch> v_good_matches_01;
    for (size_t i = 0; i < vv_matches_2nn_01.size(); i++){
      // std::cout <<vv_matches_2nn_01[i][0].distance << " vs " << vv_matches_2nn_01[i][1].distance << std::endl;
      if (vv_matches_2nn_01[i][0].distance < 0.8*vv_matches_2nn_01[i][1].distance) {
        v_good_matches_01.push_back(vv_matches_2nn_01[i][0]);
      }
    }
    cv::Mat m_output;
    std::cout << v_good_matches_01.size() << std::endl;
    cv::drawMatches(pm_images.first,v_kpts0,pm_images.second,v_kpts1,v_good_matches_01,m_output);
    cv::imshow("matches-by-tfeat", m_output);
    // cv::imwrite("matches-by-tfeat.png", m_output);
    cv::waitKey(0);
  }
  return 0;
}
