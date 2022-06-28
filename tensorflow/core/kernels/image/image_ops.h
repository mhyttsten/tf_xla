/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_IMAGE_IMAGE_OPS_H_
#define TENSORFLOW_CORE_KERNELS_IMAGE_IMAGE_OPS_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSimagePSimage_opsDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSimage_opsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSimagePSimage_opsDTh() {
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


// See docs in ../ops/image_ops.cc.

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace generator {

enum Interpolation { NEAREST, BILINEAR };
enum Mode { FILL_REFLECT, FILL_WRAP, FILL_CONSTANT, FILL_NEAREST };

using Eigen::array;
using Eigen::DenseIndex;

// Follow scipy's implementation
// https://github.com/scipy/scipy/blob/master/scipy/ndimage/src/ni_interpolation.c
template <typename Device, Mode M>
struct MapCoordinate {
  float operator()(const float out_coord, const DenseIndex len);
};

template <typename Device>
struct MapCoordinate<Device, Mode::FILL_REFLECT> {
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE float operator()(const float out_coord,
                                                         const DenseIndex len) {
    // Reflect [abcd] to [dcba|abcd|dcba].
    float in_coord = out_coord;
    if (in_coord < 0) {
      if (len <= 1) {
        in_coord = 0;
      } else {
        const DenseIndex sz2 = 2 * len;
        if (in_coord < sz2) {
          in_coord = sz2 * static_cast<DenseIndex>(-in_coord / sz2) + in_coord;
        }
        in_coord = (in_coord < -len) ? in_coord + sz2 : -in_coord - 1;
      }
    } else if (in_coord > len - 1) {
      if (len <= 1) {
        in_coord = 0;
      } else {
        const DenseIndex sz2 = 2 * len;
        in_coord -= sz2 * static_cast<DenseIndex>(in_coord / sz2);
        if (in_coord >= len) {
          in_coord = sz2 - in_coord - 1;
        }
      }
    }
    // clamp is necessary because when out_coord = 3.5 and len = 4,
    // in_coord = 3.5 and will be rounded to 4 in nearest interpolation.
    return Eigen::internal::scalar_clamp_op<float>(0.0f, len - 1)(in_coord);
  }
};

template <typename Device>
struct MapCoordinate<Device, Mode::FILL_WRAP> {
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE float operator()(const float out_coord,
                                                         const DenseIndex len) {
    // Wrap [abcd] to [abcd|abcd|abcd].
    float in_coord = out_coord;
    if (in_coord < 0) {
      if (len <= 1) {
        in_coord = 0;
      } else {
        const DenseIndex sz = len - 1;
        in_coord += len * (static_cast<DenseIndex>(-in_coord / sz) + 1);
      }
    } else if (in_coord > len - 1) {
      if (len <= 1) {
        in_coord = 0;
      } else {
        const DenseIndex sz = len - 1;
        in_coord -= len * static_cast<DenseIndex>(in_coord / sz);
      }
    }
    // clamp is necessary because when out_coord = -0.5 and len = 4,
    // in_coord = 3.5 and will be rounded to 4 in nearest interpolation.
    return Eigen::internal::scalar_clamp_op<float>(0.0f, len - 1)(in_coord);
  }
};

template <typename Device>
struct MapCoordinate<Device, Mode::FILL_CONSTANT> {
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE float operator()(const float out_coord,
                                                         const DenseIndex len) {
    return out_coord;
  }
};

template <typename Device>
struct MapCoordinate<Device, Mode::FILL_NEAREST> {
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE float operator()(const float out_coord,
                                                         const DenseIndex len) {
    return Eigen::internal::scalar_clamp_op<float>(0.0f, len - 1)(out_coord);
  }
};

template <typename Device, typename T, Mode M>
class ProjectiveGenerator {
 private:
  typename TTypes<T, 4>::ConstTensor input_;
  typename TTypes<float>::ConstMatrix transforms_;
  const Interpolation interpolation_;
  const T fill_value_;

 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  ProjectiveGenerator(typename TTypes<T, 4>::ConstTensor input,
                      typename TTypes<float>::ConstMatrix transforms,
                      const Interpolation interpolation, const T fill_value)
      : input_(input),
        transforms_(transforms),
        interpolation_(interpolation),
        fill_value_(fill_value) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSimage_opsDTh mht_0(mht_0_v, 306, "", "./tensorflow/core/kernels/image/image_ops.h", "ProjectiveGenerator");
}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T
  operator()(const array<DenseIndex, 4>& coords) const {
    const int64_t output_y = coords[1];
    const int64_t output_x = coords[2];
    const float* transform =
        transforms_.dimension(0) == 1
            ? transforms_.data()
            : &transforms_.data()[transforms_.dimension(1) * coords[0]];
    float projection = transform[6] * output_x + transform[7] * output_y + 1.f;
    if (projection == 0) {
      // Return the fill value for infinite coordinates,
      // which are outside the input image
      return fill_value_;
    }
    const float input_x =
        (transform[0] * output_x + transform[1] * output_y + transform[2]) /
        projection;
    const float input_y =
        (transform[3] * output_x + transform[4] * output_y + transform[5]) /
        projection;

    // Map out-of-boundary input coordinates to in-boundary based on fill_mode.
    auto map_functor = MapCoordinate<Device, M>();
    const float x = map_functor(input_x, input_.dimension(2));
    const float y = map_functor(input_y, input_.dimension(1));

    const DenseIndex batch = coords[0];
    const DenseIndex channels = coords[3];
    switch (interpolation_) {
      case NEAREST:
        return nearest_interpolation(batch, y, x, channels, fill_value_);
      case BILINEAR:
        return bilinear_interpolation(batch, y, x, channels, fill_value_);
    }
    // Unreachable; ImageProjectiveTransform only uses INTERPOLATION_NEAREST
    // or INTERPOLATION_BILINEAR.
    return fill_value_;
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T
  nearest_interpolation(const DenseIndex batch, const float y, const float x,
                        const DenseIndex channel, const T fill_value) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSimage_opsDTh mht_1(mht_1_v, 352, "", "./tensorflow/core/kernels/image/image_ops.h", "nearest_interpolation");

    return read_with_fill_value(batch, DenseIndex(std::round(y)),
                                DenseIndex(std::round(x)), channel, fill_value);
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T
  bilinear_interpolation(const DenseIndex batch, const float y, const float x,
                         const DenseIndex channel, const T fill_value) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSimage_opsDTh mht_2(mht_2_v, 362, "", "./tensorflow/core/kernels/image/image_ops.h", "bilinear_interpolation");

    const float y_floor = std::floor(y);
    const float x_floor = std::floor(x);
    const float y_ceil = y_floor + 1;
    const float x_ceil = x_floor + 1;
    // f(x, y_floor) = (x_ceil - x) / (x_ceil - x_floor) * f(x_floor, y_floor)
    //               + (x - x_floor) / (x_ceil - x_floor) * f(x_ceil, y_floor)
    const float value_yfloor =
        (x_ceil - x) * static_cast<float>(read_with_fill_value(
                           batch, DenseIndex(y_floor), DenseIndex(x_floor),
                           channel, fill_value)) +
        (x - x_floor) * static_cast<float>(read_with_fill_value(
                            batch, DenseIndex(y_floor), DenseIndex(x_ceil),
                            channel, fill_value));
    // f(x, y_ceil) = (x_ceil - x) / (x_ceil - x_floor) * f(x_floor, y_ceil)
    //              + (x - x_floor) / (x_ceil - x_floor) * f(x_ceil, y_ceil)
    const float value_yceil =
        (x_ceil - x) * static_cast<float>(read_with_fill_value(
                           batch, DenseIndex(y_ceil), DenseIndex(x_floor),
                           channel, fill_value)) +
        (x - x_floor) * static_cast<float>(read_with_fill_value(
                            batch, DenseIndex(y_ceil), DenseIndex(x_ceil),
                            channel, fill_value));
    // f(x, y) = (y_ceil - y) / (y_ceil - y_floor) * f(x, y_floor)
    //         + (y - y_floor) / (y_ceil - y_floor) * f(x, y_ceil)
    return T((y_ceil - y) * value_yfloor + (y - y_floor) * value_yceil);
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T read_with_fill_value(
      const DenseIndex batch, const DenseIndex y, const DenseIndex x,
      const DenseIndex channel, const T fill_value) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSimage_opsDTh mht_3(mht_3_v, 395, "", "./tensorflow/core/kernels/image/image_ops.h", "read_with_fill_value");

    // batch and channel must be correct, because they are passed unchanged from
    // the input.
    return (0 <= y && y < input_.dimension(1) && 0 <= x &&
            x < input_.dimension(2))
               ? input_(array<DenseIndex, 4>{batch, y, x, channel})
               : fill_value;
  }
};

}  // end namespace generator

namespace functor {

using generator::Interpolation;
using generator::Mode;
using generator::ProjectiveGenerator;

template <typename Device, typename T>
struct FillProjectiveTransform {
  typedef typename TTypes<T, 4>::Tensor OutputType;
  typedef typename TTypes<T, 4>::ConstTensor InputType;
  typedef typename TTypes<float, 2>::ConstTensor TransformsType;
  const Interpolation interpolation;

  explicit FillProjectiveTransform(Interpolation interpolation)
      : interpolation(interpolation) {}

  EIGEN_ALWAYS_INLINE
  void operator()(const Device& device, OutputType* output,
                  const InputType& images, const TransformsType& transform,
                  const Mode fill_mode, const T fill_value) const {
    switch (fill_mode) {
      case Mode::FILL_REFLECT:
        output->device(device) =
            output->generate(ProjectiveGenerator<Device, T, Mode::FILL_REFLECT>(
                images, transform, interpolation, fill_value));
        break;
      case Mode::FILL_WRAP:
        output->device(device) =
            output->generate(ProjectiveGenerator<Device, T, Mode::FILL_WRAP>(
                images, transform, interpolation, fill_value));
        break;
      case Mode::FILL_CONSTANT:
        output->device(device) = output->generate(
            ProjectiveGenerator<Device, T, Mode::FILL_CONSTANT>(
                images, transform, interpolation, fill_value));
        break;
      case Mode::FILL_NEAREST:
        output->device(device) =
            output->generate(ProjectiveGenerator<Device, T, Mode::FILL_NEAREST>(
                images, transform, interpolation, fill_value));
        break;
    }
  }
};

}  // end namespace functor

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_IMAGE_IMAGE_OPS_H_
