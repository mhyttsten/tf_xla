#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <string>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <unistd.h>
class MHTracer_DTPStensorflowPScorePSkernelsPSquantized_resize_bilinear_op_testDTcc {
public:
   std::string _s;
   int _indent = 0;
   std::string _functionName;
   bool _isFile = false;
   std::string _fileName;
   std::string _envMHIndent;
   int _lineNumber;
   bool _filtered = false;
   bool _otherThread = false;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_resize_bilinear_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
      _functionName = functionName;
      _lineNumber = lineNumber;

      // Check if tracing is enabled
      const char* env_path = std::getenv("PATH");
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_ENABLE") == std::string::npos) {
         return;
      }
      // Should we trace of filter?
      const char* env_filter = std::getenv("MHTRACER_FILTER");
      if (env_filter != nullptr) {
         std::string sfilter = std::string(env_filter);
         std::string sLineNumber = std::to_string(lineNumber);
         while (true) {
            std::size_t ioE = sfilter.find(";");
            if (sfilter.size() == 0) {
               break;
            }
            std::string cfs = sfilter.substr(0, ioE);
            std::size_t ioFileName = cfs.find("|");
            std::string fFileName  = cfs.substr(0, ioFileName);
            std::size_t ioFunctionName = cfs.find("|", ioFileName+1);
            std::string fFunctionName  = cfs.substr(ioFileName+1, ioFunctionName-ioFileName-1);
            std::string fLineNumber    = cfs.substr(ioFunctionName+1, cfs.size()-ioFunctionName-1);

            if (  (fFileName == "*" || fFileName == fileName)
               && (fFunctionName == "*" || fFunctionName == functionName)
               && (fLineNumber == "*" || fLineNumber == sLineNumber)) {
              _filtered = true;
               return;
            }

            if (ioE == std::string::npos) {
               sfilter = "";
            } else {
               sfilter = sfilter.substr(ioE+1, sfilter.size()-ioE-1);
            }
         }
      }

      // Create log string
      std::string ostr;

      // Assign indent spaces (tied to PID and TID)
      pid_t pid = getpid();
      std::thread::id tid = std::this_thread::get_id();
      std::stringstream pid_dash_tid_ss;
      pid_dash_tid_ss << pid << "-" << tid;
      std::string pid_dash_tid_str = pid_dash_tid_ss.str();
      _envMHIndent = "MHTRACER_INDENT_";
      char* env_indent = std::getenv(_envMHIndent.c_str());
      if (env_indent != nullptr) {
         _indent = std::stoi(std::string(env_indent));
      }
      _s.assign(_indent, ' ');

      // Check that reporting matches pid/tid
      const char* env_pid_dash_tid = std::getenv("MHTRACER_PID_DASH_TID");
      if (env_pid_dash_tid != nullptr) {
         std::string env_pid_dash_tid_str(env_pid_dash_tid);
         if (env_pid_dash_tid_str != pid_dash_tid_str) {
            _otherThread = true;
         }
      }
      else {  // PID-THREAD not set, set it for the first time (starter thread)
         setenv("MHTRACER_PID_DASH_TID", pid_dash_tid_str.c_str(), 1);
      }

      std::string paramStr;
      for (int i=0; i < params.size(); i++) {
         auto e = params[i];
         while (e.find("\n") != std::string::npos) {
            size_t pos = e.find("\n");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<NL>");
         }
         while (e.find("[") != std::string::npos) {
            size_t pos = e.find("[");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<LB>");
         }
         while (e.find("]") != std::string::npos) {
            size_t pos = e.find("]");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<RB>");
         }
         paramStr += e;
         if ((i+1) < params.size()) {
            paramStr += ", ";
         }
      }

      const char* env_dont_print_pid_dash_tid = std::getenv("MHTRACER_DONT_PRINT_PID_DASH_TID");
      if (env_dont_print_pid_dash_tid != nullptr) {
         pid_dash_tid_str = "";
      }
      if (_otherThread) {
         functionName = "MHOT_" + functionName;
      }
      ostr += _s + functionName + 
         + " [1]"
         + " [" + prefix + "]"
         + " [" + paramStr + "]"
         + " [" + pid_dash_tid_str + " "
         +    std::to_string(lineNumber)
         +    " @ " + fileName + "]\n";

      // Log to file
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_USEFILE") != std::string::npos) {
         _isFile = true;
         _fileName = "/tmp/mhtracer_" + pid_dash_tid_str + ".log";
         std::ofstream os;
         os.open(_fileName, std::ofstream::out | std::ofstream::app);
         os << ostr << "";
         os.close();
      }
      // Log to stdout
      else {
         std::cout << ostr << "";
      }

      // Increment indent spaces
      if (_otherThread) {
         return;
      }
      _indent += 3;
      setenv(_envMHIndent.c_str(), std::to_string(_indent).c_str(), 1);
   }
   ~MHTracer_DTPStensorflowPScorePSkernelsPSquantized_resize_bilinear_op_testDTcc() {
      // Check if tracing is enabled
      char* env_path = std::getenv("PATH");
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_ENABLE") == std::string::npos) {
         return;
      }

      // Don't update indent if tracing was filtered or from another thread
      if (_filtered || _otherThread) {
         return;
      }

      _indent -= 3;
      setenv(_envMHIndent.c_str(), std::to_string(_indent).c_str(), 1);
   }
};

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#define EIGEN_USE_THREADS

#include <vector>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/core/common_runtime/gradients.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

namespace {
constexpr const float RESIZE_VAL_TOLERANCE = 1.0e-8;

template <typename T>
Tensor BuildTensor(const int batch_size, const int height, const int width,
                   const int channels, const float ratio, const float min,
                   const float max) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_resize_bilinear_op_testDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/kernels/quantized_resize_bilinear_op_test.cc", "BuildTensor");

  Tensor tensor(DataTypeToEnum<T>::value,
                TensorShape({batch_size, height, width, channels}));
  for (int64_t i = 0; i < tensor.NumElements(); ++i) {
    tensor.flat<T>()(i) =
        FloatToQuantized<T>(static_cast<float>(i) / ratio, min, max);
  }
  return tensor;
}

template <>
Tensor BuildTensor<float>(const int batch_size, const int height,
                          const int width, const int channels,
                          const float ratio, const float min, const float max) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_resize_bilinear_op_testDTcc mht_1(mht_1_v, 226, "", "./tensorflow/core/kernels/quantized_resize_bilinear_op_test.cc", "BuildTensor<float>");

  Tensor tensor(DT_FLOAT, TensorShape({batch_size, height, width, channels}));
  for (int64_t i = 0; i < tensor.NumElements(); ++i) {
    tensor.flat<float>()(i) = static_cast<float>(i) / ratio;
  }
  return tensor;
}

float CalculateResizeScale(int64_t in_size, int64_t out_size,
                           bool align_corners) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_resize_bilinear_op_testDTcc mht_2(mht_2_v, 238, "", "./tensorflow/core/kernels/quantized_resize_bilinear_op_test.cc", "CalculateResizeScale");

  return (align_corners && out_size > 1)
             ? (in_size - 1) / static_cast<float>(out_size - 1)
             : in_size / static_cast<float>(out_size);
}

inline std::tuple<int64_t, int64_t, float> GetReferenceWeight(
    const bool half_pixel_centers, const int64_t out_size,
    const int64_t in_size, const int step, const int index, const float scale) {
  const float in = half_pixel_centers
                       ? (static_cast<float>(index) + 0.5f) * scale - 0.5f
                       : index * scale;
  const float in_f = std::floor(in);
  const int64_t lower =
      std::max(static_cast<int64_t>(in_f), static_cast<int64_t>(0));
  const int64_t upper =
      std::min(static_cast<int64_t>(std::ceil(in)), in_size - 1);
  return std::make_tuple(lower * step, upper * step, in - in_f);
}

template <typename T>
T ComputeLerpReference(const T in_top_left, const T in_top_right,
                       const T in_bottom_left, const T in_bottom_right,
                       const float x_lerp, const float y_lerp, const float min,
                       const float max) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_resize_bilinear_op_testDTcc mht_3(mht_3_v, 265, "", "./tensorflow/core/kernels/quantized_resize_bilinear_op_test.cc", "ComputeLerpReference");

  const float top_left = QuantizedToFloat<T>(in_top_left, min, max);
  const float top_right = QuantizedToFloat<T>(in_top_right, min, max);
  const float bottom_left = QuantizedToFloat<T>(in_bottom_left, min, max);
  const float bottom_right = QuantizedToFloat<T>(in_bottom_right, min, max);
  const float top = top_left + (top_right - top_left) * x_lerp;
  const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
  const float out = top + (bottom - top) * y_lerp;
  return FloatToQuantized<T>(out, min, max);
}

template <>
float ComputeLerpReference<float>(const float in_top_left,
                                  const float in_top_right,
                                  const float in_bottom_left,
                                  const float in_bottom_right,
                                  const float x_lerp, const float y_lerp,
                                  const float min, const float max) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_resize_bilinear_op_testDTcc mht_4(mht_4_v, 285, "", "./tensorflow/core/kernels/quantized_resize_bilinear_op_test.cc", "ComputeLerpReference<float>");

  const float top = in_top_left + (in_top_right - in_top_left) * x_lerp;
  const float bottom =
      in_bottom_left + (in_bottom_right - in_bottom_left) * x_lerp;
  return top + (bottom - top) * y_lerp;
}

template <typename T>
T CalcReferenceResizedVal(const T* image_data, const bool half_pixel_centers,
                          const int batch_size, const int64_t in_height,
                          const int64_t in_width, const int64_t out_height,
                          const int64_t out_width, const int channels,
                          const float height_scale, const float width_scale,
                          const float min, const float max, const int b,
                          const int64_t x, const int64_t y, const int c) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_resize_bilinear_op_testDTcc mht_5(mht_5_v, 302, "", "./tensorflow/core/kernels/quantized_resize_bilinear_op_test.cc", "CalcReferenceResizedVal");

  const std::tuple<int64_t, int64_t, float> x_weight = GetReferenceWeight(
      half_pixel_centers, out_width, in_width, channels, x, width_scale);
  const std::tuple<int64_t, int64_t, float> y_weight = GetReferenceWeight(
      half_pixel_centers, out_height, in_height, 1, y, height_scale);

  const int64_t in_row_size = in_width * channels;
  const int64_t in_batch_num_values = in_height * in_row_size;

  const int y_lower_index =
      b * in_batch_num_values + std::get<0>(y_weight) * in_row_size;
  const int y_upper_index =
      b * in_batch_num_values + std::get<1>(y_weight) * in_row_size;

  const int64_t xs_lower = std::get<0>(x_weight);
  const int64_t xs_upper = std::get<1>(x_weight);
  const float xs_lerp = std::get<2>(x_weight);
  const float ys_lerp = std::get<2>(y_weight);
  const float top_left = image_data[y_lower_index + xs_lower + c];
  const float top_right = image_data[y_lower_index + xs_upper + c];
  const float bottom_left = image_data[y_upper_index + xs_lower + c];
  const float bottom_right = image_data[y_upper_index + xs_upper + c];
  const float val =
      ComputeLerpReference<T>(top_left, top_right, bottom_left, bottom_right,
                              xs_lerp, ys_lerp, min, max);
  return val;
}

template <typename T>
void CheckTensorValue(const T* in_data, const T* out_data, const int batch_size,
                      const int64_t in_height, const int64_t in_width,
                      const int64_t out_height, const int64_t out_width,
                      const int channels, const bool align_corners,
                      const bool half_pixel_centers, const float min,
                      const float max, const float tolerance,
                      const bool relative) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_resize_bilinear_op_testDTcc mht_6(mht_6_v, 340, "", "./tensorflow/core/kernels/quantized_resize_bilinear_op_test.cc", "CheckTensorValue");

  const int64_t out_row_size = out_width * channels;
  const float height_scale =
      CalculateResizeScale(in_height, out_height, align_corners);
  const float width_scale =
      CalculateResizeScale(in_width, out_width, align_corners);

  for (int b = 0; b < batch_size; ++b) {
    for (int64_t y = 0; y < out_height; ++y) {
      for (int64_t x = 0; x < out_width; ++x) {
        for (int c = 0; c < channels; ++c) {
          const T ref_qval = CalcReferenceResizedVal<T>(
              in_data, half_pixel_centers, batch_size, in_height, in_width,
              out_height, out_width, channels, height_scale, width_scale, min,
              max, b, x, y, c);
          const T qval =
              out_data[(b * out_height + y) * out_row_size + x * channels + c];
          const float ref_val = QuantizedToFloat<T>(ref_qval, min, max);
          const float val = QuantizedToFloat<T>(qval, min, max);
          if (!relative) {
            const int q_tolerance = std::round(tolerance);
            EXPECT_TRUE(std::abs(static_cast<int32>(ref_qval) -
                                 static_cast<int32>(qval)) <= q_tolerance)
                << "ref = " << ref_val << ", val = " << val << ", " << b << ", "
                << y << ", " << x << ", " << c << ", qval = " << qval
                << ", ref qval = " << ref_qval << ", " << q_tolerance;
          } else {
            const float rel_tolerance = std::max(ref_val, 1.0f) * tolerance;
            EXPECT_NEAR(ref_val, val, rel_tolerance)
                << "ref = " << ref_val << ", val = " << val << ", " << b << ", "
                << y << ", " << x << ", " << c << ", ref qval = " << qval;
          }
        }
      }
    }
  }
}

void TestResizeBilinear(const Tensor& image_tensor, const DataType dt,
                        const Input::Initializer& new_size,
                        const bool show_time, const int64_t iterations,
                        const float min, const float max,
                        const bool half_pixel_centers,
                        std::vector<Tensor>* outputs) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_resize_bilinear_op_testDTcc mht_7(mht_7_v, 386, "", "./tensorflow/core/kernels/quantized_resize_bilinear_op_test.cc", "TestResizeBilinear");

  Scope root = Scope::NewRootScope();

  Output placeholder = ops::Placeholder(root.WithOpName("placeholder"), dt);
  Output size = ops::Const<int32>(root.WithOpName("size"), new_size);
  Output in_min = ops::Const<float>(root.WithOpName("min"), min);
  Output in_max = ops::Const<float>(root.WithOpName("max"), max);

  ops::QuantizedResizeBilinear qrb = ops::QuantizedResizeBilinear(
      root.WithOpName("qrb"), placeholder, size, in_min, in_max,
      ops::QuantizedResizeBilinear::HalfPixelCenters(half_pixel_centers));

  TF_EXPECT_OK(root.status());

  ClientSession session(root);

  int64_t total_duration = 0;
  outputs->clear();

  for (int i = 0; i < iterations; ++i) {
    const int64_t start_time = Env::Default()->NowMicros();
    TF_EXPECT_OK(session.Run({{placeholder, image_tensor}},
                             {qrb.resized_images, qrb.out_min, qrb.out_max},
                             outputs));
    const int64_t end_time = Env::Default()->NowMicros();
    total_duration += end_time - start_time;
  }
  const int64_t one_run_duration = total_duration / iterations;

  const int64_t num_ops = outputs->at(0).NumElements();

  const double million_ops_per_second =
      (iterations * num_ops) / static_cast<double>(total_duration);

  if (show_time) {
    LOG(INFO) << "Time resize bilinear: "
              << TensorShape(image_tensor.shape()).DebugString()
              << ": iterations=" << iterations
              << ", MOps/s=" << million_ops_per_second
              << ", one_run_duration=" << one_run_duration
              << ", total_duration=" << total_duration;
  }
}

}  // namespace

void TestResizeBilinearOneDim() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_resize_bilinear_op_testDTcc mht_8(mht_8_v, 435, "", "./tensorflow/core/kernels/quantized_resize_bilinear_op_test.cc", "TestResizeBilinearOneDim");

  constexpr float TOLERANCE = 1.0e-5;
  constexpr int IN_WIDTH = 128;
  constexpr int OUT_WIDTH = 256;
  constexpr float MIN = 0.0f;
  constexpr float MAX = 256.0f;
  constexpr float SCALE = static_cast<float>(IN_WIDTH) / OUT_WIDTH;
  Tensor image_quantized_tensor(DT_QINT32, TensorShape({1, 1, IN_WIDTH, 1}));

  for (int64_t i = 0; i < image_quantized_tensor.NumElements(); ++i) {
    image_quantized_tensor.flat<qint32>()(i) =
        FloatToQuantized<qint32>(static_cast<float>(i), MIN, MAX);
  }

  std::vector<Tensor> outputs;
  TestResizeBilinear(image_quantized_tensor, DT_QINT32, {1, OUT_WIDTH}, false,
                     1, MIN, MAX, false, &outputs);
  ASSERT_EQ(3, outputs.size());
  ASSERT_EQ(OUT_WIDTH, outputs.at(0).NumElements());
  ASSERT_EQ(4, outputs.at(0).shape().dims());
  ASSERT_EQ(OUT_WIDTH, outputs.at(0).shape().dim_size(2));

  // Manual value testing
  for (int64_t i = 0; i < outputs.at(0).NumElements(); ++i) {
    const float resized_image_val =
        QuantizedToFloat<qint32>(outputs.at(0).flat<qint32>()(i), MIN, MAX);
    float expected_val = 0.0f;
    if (i == 0 || i == outputs.at(0).NumElements() - 1 || i % 2 == 0) {
      expected_val = QuantizedToFloat<qint32>(
          image_quantized_tensor.flat<qint32>()(i / 2), MIN, MAX);
    } else {
      const float image_val0 = QuantizedToFloat<qint32>(
          image_quantized_tensor.flat<qint32>()(i / 2), MIN, MAX);
      const float image_val1 = QuantizedToFloat<qint32>(
          image_quantized_tensor.flat<qint32>()(i / 2 + 1), MIN, MAX);
      expected_val = (image_val0 + image_val1) * SCALE;
    }
    VLOG(1) << "(" << i << ") " << expected_val << ", " << resized_image_val;
    EXPECT_NEAR(expected_val, resized_image_val, RESIZE_VAL_TOLERANCE)
        << expected_val << ", " << resized_image_val;
  }

  // Value testing with reference implementation
  CheckTensorValue<qint32>(image_quantized_tensor.flat<qint32>().data(),
                           outputs.at(0).flat<qint32>().data(),
                           /*batch_size=*/1,
                           /*in_height=*/IN_WIDTH,
                           /*in_width=*/1,
                           /*out_height=*/OUT_WIDTH,
                           /*out_width=*/1,
                           /*channels=*/1,
                           /*align_corners=*/false,
                           /*half_pixel_centers=*/false, MIN, MAX, TOLERANCE,
                           true);
}

template <typename T>
void RunTestResizeBilinearTwoDims(int batch_size, int in_height, int in_width,
                                  int out_height, int out_width, int channels,
                                  float tolerance, bool relative,
                                  const bool half_pixel_centers) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_resize_bilinear_op_testDTcc mht_9(mht_9_v, 498, "", "./tensorflow/core/kernels/quantized_resize_bilinear_op_test.cc", "RunTestResizeBilinearTwoDims");

  constexpr float RATIO = 100.0f;
  const float min = 0.0f;
  const float max = batch_size * in_height * in_width * channels / RATIO;

  const Tensor image_quantized_tensor = BuildTensor<T>(
      batch_size, in_height, in_width, channels, RATIO, min, max);

  std::vector<Tensor> outputs;
  TestResizeBilinear(image_quantized_tensor, DataTypeToEnum<T>::value,
                     {out_height, out_width}, false, 1, min, max,
                     half_pixel_centers, &outputs);
  CheckTensorValue<T>(
      image_quantized_tensor.flat<T>().data(), outputs.at(0).flat<T>().data(),
      batch_size, in_height, in_width, out_height, out_width, channels,
      /*align_corners=*/false,
      /*half_pixel_centers=*/half_pixel_centers, min, max, tolerance, relative);
}

template <typename T>
void RunBenchmarkResizeBilinearTwoDims(int batch_size, int in_height,
                                       int in_width, int out_height,
                                       int out_width, int channels,
                                       int iteration,
                                       const bool half_pixel_centers) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_resize_bilinear_op_testDTcc mht_10(mht_10_v, 525, "", "./tensorflow/core/kernels/quantized_resize_bilinear_op_test.cc", "RunBenchmarkResizeBilinearTwoDims");

  constexpr float RATIO = 100.0f;
  const float min = 0.0f;
  const float max = batch_size * in_height * in_width * channels / RATIO;

  const Tensor image_quantized_tensor = BuildTensor<T>(
      batch_size, in_height, in_width, channels, RATIO, min, max);

  std::vector<Tensor> outputs;
  TestResizeBilinear(image_quantized_tensor, DataTypeToEnum<T>::value,
                     {out_height, out_width}, true, iteration, min, max, false,
                     &outputs);
}

template <typename T>
void TestResizeBilinearTwoDimsType(const float tolerance, const bool relative,
                                   const bool half_pixel_centers) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_resize_bilinear_op_testDTcc mht_11(mht_11_v, 544, "", "./tensorflow/core/kernels/quantized_resize_bilinear_op_test.cc", "TestResizeBilinearTwoDimsType");

  RunTestResizeBilinearTwoDims<T>(1, 1, 1, 1, 1, 1, tolerance, relative,
                                  half_pixel_centers);
  RunTestResizeBilinearTwoDims<T>(1, 1, 128, 1, 256, 1, tolerance, relative,
                                  half_pixel_centers);
  RunTestResizeBilinearTwoDims<T>(1, 128, 1, 256, 1, 1, tolerance, relative,
                                  half_pixel_centers);
  RunTestResizeBilinearTwoDims<T>(1, 128, 128, 256, 256, 1, tolerance, relative,
                                  half_pixel_centers);
  RunTestResizeBilinearTwoDims<T>(1, 256, 256, 128, 128, 1, tolerance, relative,
                                  half_pixel_centers);
  RunTestResizeBilinearTwoDims<T>(1, 1, 128, 1, 256, 2, tolerance, relative,
                                  half_pixel_centers);
  RunTestResizeBilinearTwoDims<T>(1, 128, 1, 256, 1, 2, tolerance, relative,
                                  half_pixel_centers);
  RunTestResizeBilinearTwoDims<T>(1, 128, 128, 256, 256, 2, tolerance, relative,
                                  half_pixel_centers);
  RunTestResizeBilinearTwoDims<T>(1, 256, 256, 128, 128, 2, tolerance, relative,
                                  half_pixel_centers);
  RunTestResizeBilinearTwoDims<T>(1, 1, 16, 1, 32, 3, tolerance, relative,
                                  half_pixel_centers);
  RunTestResizeBilinearTwoDims<T>(1, 1, 128, 1, 256, 3, tolerance, relative,
                                  half_pixel_centers);
  RunTestResizeBilinearTwoDims<T>(1, 128, 128, 256, 256, 3, tolerance, relative,
                                  half_pixel_centers);
  RunTestResizeBilinearTwoDims<T>(1, 256, 256, 128, 128, 3, tolerance, relative,
                                  half_pixel_centers);
}

void TestResizeBilinearTwoDims() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_resize_bilinear_op_testDTcc mht_12(mht_12_v, 576, "", "./tensorflow/core/kernels/quantized_resize_bilinear_op_test.cc", "TestResizeBilinearTwoDims");

  for (const bool half_pixel_centers : {false, true}) {
    TestResizeBilinearTwoDimsType<quint8>(1.0f, false, half_pixel_centers);
    TestResizeBilinearTwoDimsType<qint32>(1.0e-5, true, half_pixel_centers);
    TestResizeBilinearTwoDimsType<float>(1.0e-5, true, half_pixel_centers);
  }
}

template <typename T>
void RunBenchmarkResizeBilinearTwoDimsType() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_resize_bilinear_op_testDTcc mht_13(mht_13_v, 588, "", "./tensorflow/core/kernels/quantized_resize_bilinear_op_test.cc", "RunBenchmarkResizeBilinearTwoDimsType");

  constexpr int ITER = 100;
  RunBenchmarkResizeBilinearTwoDims<T>(1, 1, 1, 2, 2, 1, ITER, false);
  RunBenchmarkResizeBilinearTwoDims<T>(1, 128, 128, 256, 256, 1, ITER, false);
  RunBenchmarkResizeBilinearTwoDims<T>(1, 128, 128, 256, 256, 3, ITER, false);
  RunBenchmarkResizeBilinearTwoDims<T>(1, 64, 64, 128, 128, 2, ITER, false);
  RunBenchmarkResizeBilinearTwoDims<T>(1, 32, 32, 64, 64, 16, ITER, false);
}

void RunBenchmarkResizeBilinearTwoDims() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_resize_bilinear_op_testDTcc mht_14(mht_14_v, 600, "", "./tensorflow/core/kernels/quantized_resize_bilinear_op_test.cc", "RunBenchmarkResizeBilinearTwoDims");

  LOG(INFO) << "Benchmark quint8";
  RunBenchmarkResizeBilinearTwoDimsType<quint8>();
  LOG(INFO) << "Benchmark qint32";
  RunBenchmarkResizeBilinearTwoDimsType<qint32>();
  LOG(INFO) << "Benchmark float";
  RunBenchmarkResizeBilinearTwoDimsType<float>();
}

}  // namespace tensorflow

#define RUN_TEST(t) \
  TEST(QuantizationResizeBilinearTest, t) { tensorflow::t(); }

RUN_TEST(TestResizeBilinearOneDim);
RUN_TEST(TestResizeBilinearTwoDims);

#if defined(__ANDROID__)

RUN_TEST(RunBenchmarkResizeBilinearTwoDims);

#endif  // __ANDROID__

int main(int argc, char** argv) {
  // On Linux, add: absl::SetFlag(&FLAGS_logtostderr, true);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
