#include <torch/script.h> // One-stop header.
#include <torch/torch.h>

#include <iostream>
#include <vector>
#include <memory>
#include <cmath>

#include <opencv2/opencv.hpp>

#define PI 3.14159265


void process(const std::shared_ptr<torch::jit::script::Module>& module,
             const cv::Mat& m_image,
             cv::Mat& m_descriptors,
             std::vector<cv::KeyPoint>& v_kpts);

std::vector<cv::Mat> extractPatches(
                      const cv::Mat& m_image,
                      const std::vector<cv::KeyPoint>& v_kpts,
                      float f_mag_factor = 3.0);

int main(int argc, char* argv[]) {

  if (argc != 4){
    std::cout << "usage : this.out [/path/to/model.pt] [/path/to/image0] [/path/to/image1]";
    std::cout << std::endl;
    return -1;
  }


  // Loading your model
  const std::string s_model_name = argv[1];
  std::cout << " >>> Loading " << s_model_name;
  std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(s_model_name);

  assert(module != nullptr);
  std::cout << " ... Done! " << std::endl;;

  // Loading your images
  const std::string s_image_name0 = argv[2];
  const std::string s_image_name1 = argv[3];

  std::cout << " >>> Loading " << s_image_name0 << " and " << s_image_name1;
  cv::Mat m_image0 = cv::imread(s_image_name0, 0);
  cv::Mat m_image1 = cv::imread(s_image_name1, 0);
  std::cout << " ... Done! " << std::endl;

  std::cout << " >>> Forward computation started ";
  cv::Mat m_descriptors0, m_descriptors1;
  std::vector<cv::KeyPoint> v_kpts0, v_kpts1;

  process(module, m_image0, m_descriptors0, v_kpts0);
  process(module, m_image1, m_descriptors1, v_kpts1);

  std::cout << " ... Done!" << std::endl;

  cv::BFMatcher* matcher = new cv::BFMatcher(cv::NORM_L2, false);
  std::vector< std::vector<cv::DMatch> > vv_matches_2nn_01;
  matcher->knnMatch(m_descriptors0, m_descriptors1, vv_matches_2nn_01, 2);

  std::vector<cv::DMatch> v_good_matches_01;
  for (size_t i = 0; i < vv_matches_2nn_01.size(); i++){
    if (vv_matches_2nn_01[i][0].distance < 0.8*vv_matches_2nn_01[i][1].distance) {
      v_good_matches_01.push_back(vv_matches_2nn_01[i][0]);
    }
  }
  cv::Mat m_output;
  cv::drawMatches(m_image0,v_kpts0,m_image1,v_kpts1,v_good_matches_01,m_output);
  cv::imshow("matches-by-tfeat", m_output);
  cv::imwrite("matches-by-tfeat.png", m_output);
  cv::waitKey(0);


  std::cout << " >>> Demo has finised !" << std::endl;

  return 0;
}


void process(const std::shared_ptr<torch::jit::script::Module>& module,
             const cv::Mat& m_image,
             cv::Mat& m_descriptors,
             std::vector<cv::KeyPoint>& v_kpts)
{
  cv::Ptr<cv::BRISK> brisk = cv::BRISK::create();
  cv::Mat _m_descriptors_brisk;
  std::vector<cv::KeyPoint> _v_kpts;
  brisk->detectAndCompute(m_image, cv::noArray(), _v_kpts, _m_descriptors_brisk);
  v_kpts = _v_kpts;

  std::vector<cv::Mat> vm_patches = extractPatches(m_image, v_kpts);

  // define tensor dimention you will generate from cv::Mat
  std::vector<int64_t> dims{static_cast<int64_t>(vm_patches.size()), 
                            static_cast<int64_t>(vm_patches[0].channels()), // 1
                            static_cast<int64_t>(vm_patches[0].rows), // 32
                            static_cast<int64_t>(vm_patches[0].cols)}; // 32

  cv::Mat mf_patches_concat = cv::Mat::zeros(32*vm_patches.size(), 32, CV_32FC1);
  for(size_t i=0; i < vm_patches.size(); i++) {
    cv::Mat mf_patch;
    vm_patches[i].convertTo(mf_patch, CV_32FC1);
    mf_patch.copyTo(mf_patches_concat.rowRange(i*32,(i+1)*32));
  }

  at::TensorOptions options(at::kFloat);
  at::Tensor input_tensor = torch::from_blob(mf_patches_concat.data, at::IntList(dims), options);

  at::Tensor output_tensor = module->forward({input_tensor}).toTensor().clone();

  cv::Mat m_feature(cv::Size(128, vm_patches.size()), CV_32FC1, output_tensor.data<float>());

  m_descriptors.release();
  m_descriptors = m_feature.clone();

  return;
}


std::vector<cv::Mat> extractPatches(
                      const cv::Mat& m_image,
                      const std::vector<cv::KeyPoint>& v_kpts,
                      float f_mag_factor) 
{
  /*
   * original python source code in tfeat is
   * converted to cpp
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
  const unsigned int N = 32;

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



