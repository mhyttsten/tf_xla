/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_FAKE_QUANT_OPS_FUNCTOR_H_
#define TENSORFLOW_CORE_KERNELS_FAKE_QUANT_OPS_FUNCTOR_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSfake_quant_ops_functorDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSfake_quant_ops_functorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSfake_quant_ops_functorDTh() {
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


#include <tuple>

#define EIGEN_STACK_ALLOCATION_LIMIT 0
#define EIGEN_USE_THREADS
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE float StdRound(float input) {
// On Android, std::round() isn't present, just round().
#if defined(__ANDROID__)
  return round(input);
#else
  return std::round(input);
#endif
}

namespace tensorflow {

// Gymnastics with nudged zero point is to ensure that real zero maps to
// an integer, which is required for e.g. zero-padding in convolutional layers.
// Outputs nudged_min, nudged_max, nudged_scale.
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void Nudge(
    const float min, const float max, const int quant_min, const int quant_max,
    float* nudged_min, float* nudged_max, float* scale, float* inv_scale) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfake_quant_ops_functorDTh mht_0(mht_0_v, 212, "", "./tensorflow/core/kernels/fake_quant_ops_functor.h", "Nudge");

  const float quant_min_float = static_cast<float>(quant_min);
  const float quant_max_float = static_cast<float>(quant_max);
  *scale = (max - min) / (quant_max_float - quant_min_float);
  // Re-calculate the inverse to avoid loss of precision which would result
  // from simply taking the reciprocal of *scale
  *inv_scale = (quant_max_float - quant_min_float) / (max - min);
  const float zero_point_from_min = quant_min_float - min / *scale;
  const uint16 nudged_zero_point = [zero_point_from_min, quant_min,
                                    quant_min_float, quant_max,
                                    quant_max_float] {
    if (zero_point_from_min < quant_min_float) {
      return static_cast<uint16>(quant_min);
    }
    if (zero_point_from_min > quant_max_float) {
      return static_cast<uint16>(quant_max);
    }
    return static_cast<uint16>(StdRound(zero_point_from_min));
  }();
  *nudged_min = (quant_min_float - nudged_zero_point) * (*scale);
  *nudged_max = (quant_max_float - nudged_zero_point) * (*scale);
}

template <typename T>
using ConstScalar = typename tensorflow::TTypes<T>::ConstScalar;
template <typename T>
using Scalar = typename tensorflow::TTypes<T>::Scalar;
template <typename T>
using ConstVec = typename tensorflow::TTypes<T>::ConstVec;
template <typename T>
using Vec = typename tensorflow::TTypes<T>::Vec;
template <typename T>
using ConstFlat = typename tensorflow::TTypes<T>::ConstFlat;
template <typename T>
using Flat = typename tensorflow::TTypes<T>::Flat;

// Functor called by FakeQuantWithMinMaxArgsOp to do the work.  Compiles both
// for CPU and GPU.
template <typename Device>
struct FakeQuantWithMinMaxArgsFunctor {
  void operator()(const Device& d, ConstFlat<float> inputs, const float min,
                  const float max, const int quant_min, const int quant_max,
                  Flat<float> outputs) {
    eigen_assert(min <= 0.0f && "min should be <= 0.0");
    eigen_assert(max >= 0.0f && "max should be >= 0.0");
    eigen_assert(min < max && "min should be < max");

    float nudged_min, nudged_max, nudged_scale, inv_nudged_scale;
    Nudge(min, max, quant_min, quant_max, &nudged_min, &nudged_max,
          &nudged_scale, &inv_nudged_scale);

    const float quant_zero = floor(-nudged_min * inv_nudged_scale + 0.5f);

    auto clamped = inputs.cwiseMin(nudged_max).cwiseMax(nudged_min);
    auto clamped_shifted = clamped - nudged_min;
    outputs.device(d) =
        (clamped_shifted * inv_nudged_scale - quant_zero + 0.5f).floor() *
        nudged_scale;
  }
};

// Functor called by FakeQuantWithMinMaxArgsGradientOp to do the work.  Compiles
// both for CPU and GPU.
template <typename Device>
struct FakeQuantWithMinMaxArgsGradientFunctor {
  void operator()(const Device& d, ConstFlat<float> gradients,
                  ConstFlat<float> inputs, const float min, const float max,
                  const int quant_min, const int quant_max,
                  Flat<float> backprops) {
    eigen_assert(min <= 0.0f && "min should be <= 0.0");
    eigen_assert(max >= 0.0f && "max should be >= 0.0");
    eigen_assert(min < max && "min should be < max");

    float nudged_min, nudged_max, nudged_scale, inv_nudged_scale;
    Nudge(min, max, quant_min, quant_max, &nudged_min, &nudged_max,
          &nudged_scale, &inv_nudged_scale);

    auto between_nudged_min_max =
        (inputs >= nudged_min && inputs <= nudged_max)
            .select(inputs.constant(1.0f), inputs.constant(0.0f));
    backprops.device(d) = gradients * between_nudged_min_max;
  }
};

// Functor called by FakeQuantWithMinMaxVarsOp to do the work.  Compiles both
// for CPU and GPU.
template <typename Device>
struct FakeQuantWithMinMaxVarsFunctor {
  void operator()(const Device& d, ConstFlat<float> inputs,
                  ConstScalar<float> min, ConstScalar<float> max,
                  const int quant_min, const int quant_max,
                  Flat<float> outputs) {
    const float min_val = min();
    const float max_val = max();
    // If min and max are both zero, we should just return zero.
    if (min_val == 0.0f && max_val == 0.0f) {
      outputs.device(d) = outputs.constant(0.0f);
      return;
    }
    float nudged_min, nudged_max, nudged_scale, inv_nudged_scale;
    Nudge(min_val, max_val, quant_min, quant_max, &nudged_min, &nudged_max,
          &nudged_scale, &inv_nudged_scale);

    const float quant_zero = floor(-nudged_min * inv_nudged_scale + 0.5f);
    const auto nudged_scale_repl = inputs.constant(nudged_scale);
    // const auto inv_nudged_scale_repl = inputs.constant(inv_nudged_scale);

    const auto clamped = inputs.cwiseMin(nudged_max).cwiseMax(nudged_min);
    const auto clamped_shifted = clamped - nudged_min;
    outputs.device(d) =
        (clamped_shifted / nudged_scale_repl - quant_zero + 0.5f).floor() *
        nudged_scale_repl;
  }
};

// Functor called by FakeQuantWithMinMaxVarsGradientOp to do the work.  Compiles
// both for CPU and GPU.
template <typename Device>
struct FakeQuantWithMinMaxVarsGradientFunctor {
  void operator()(const Device& d, ConstFlat<float> gradients,
                  ConstFlat<float> inputs, ConstScalar<float> min,
                  ConstScalar<float> max, const int quant_min,
                  const int quant_max, Flat<float> backprops_wrt_input,
                  Scalar<float> backprop_wrt_min,
                  Scalar<float> backprop_wrt_max) {
    const float min_val = min();
    const float max_val = max();
    // If min and max are both zero, we propagate everything to inputs.
    if (min_val == 0.0f && max_val == 0.0f) {
      backprops_wrt_input.device(d) = gradients;
      backprop_wrt_min.device(d) = backprop_wrt_min.constant(0.0f);
      backprop_wrt_max.device(d) = backprop_wrt_max.constant(0.0f);
      return;
    }
    float nudged_min, nudged_max, nudged_scale, inv_nudged_scale;
    Nudge(min_val, max_val, quant_min, quant_max, &nudged_min, &nudged_max,
          &nudged_scale, &inv_nudged_scale);

    const auto between_min_max =
        (inputs >= nudged_min && inputs <= nudged_max)
            .select(inputs.constant(1.0f), inputs.constant(0.0f));
    backprops_wrt_input.device(d) = gradients * between_min_max;

    const auto below_min =
        (inputs < nudged_min)
            .select(inputs.constant(1.0f), inputs.constant(0.0f));
    backprop_wrt_min.device(d) = (gradients * below_min).sum();

    const auto above_max =
        (inputs > nudged_max)
            .select(inputs.constant(1.0f), inputs.constant(0.0f));
    backprop_wrt_max.device(d) = (gradients * above_max).sum();
  }
};

using Index = typename tensorflow::TTypes<float>::ConstTensor::Index;

// Functor called by FakeQuantWithMinMaxVarsPerChannelOp to do the work.
// Compiles both for CPU and GPU.
//
// Already verified: inputs, outputs are of shape [b, d], min, max are of shape
// [d].
template <typename Device>
struct FakeQuantWithMinMaxVarsPerChannelFunctor {
  void operator()(const Device& d, TTypes<float>::ConstMatrix inputs,
                  ConstVec<float> min, ConstVec<float> max, const int quant_min,
                  const int quant_max, TTypes<float>::Matrix outputs) {
    for (Index i = 0; i < min.size(); ++i) {
      const float min_val = min(i);
      const float max_val = max(i);
      // If min and max are both zero, we should just return zero.
      if (min_val == 0.0f && max_val == 0.0f) {
        auto chip = outputs.chip<1>(i);
        chip.device(d) = chip.constant(0.0f);
        continue;
      }
      float nudged_min, nudged_max, nudged_scale, inv_nudged_scale;
      Nudge(min_val, max_val, quant_min, quant_max, &nudged_min, &nudged_max,
            &nudged_scale, &inv_nudged_scale);

      const float quant_zero = floor(-nudged_min * inv_nudged_scale + 0.5f);

      const auto clamped =
          inputs.chip<1>(i).cwiseMin(nudged_max).cwiseMax(nudged_min);
      const auto clamped_shifted = clamped - nudged_min;

      outputs.chip<1>(i).device(d) =
          (clamped_shifted * inv_nudged_scale - quant_zero + 0.5f).floor() *
          nudged_scale;
    }
  }
};

// Functor called by FakeQuantWithMinMaxVarsPerChannelGradientOp to do the work.
// Compiles both for CPU and GPU.
//
// Already verified: gradients, inputs, backprops_wrt_input are of shape [b, d],
// min, max, backprop_wrt_min, backprop_wrt_max are of shape [d].
template <typename Device>
struct FakeQuantWithMinMaxVarsPerChannelGradientFunctor {
  void operator()(const Device& d, TTypes<float>::ConstMatrix gradients,
                  TTypes<float>::ConstMatrix inputs, ConstVec<float> min,
                  ConstVec<float> max, const int quant_min, const int quant_max,
                  TTypes<float>::Matrix backprops_wrt_input,
                  Vec<float> backprop_wrt_min, Vec<float> backprop_wrt_max) {
    for (Index i = 0; i < min.size(); ++i) {
      const float min_val = min(i);
      const float max_val = max(i);
      const auto gradients_chip = gradients.chip<1>(i);
      const auto inputs_chip = inputs.chip<1>(i);
      // If min and max are both zero, we propagate everything to inputs.
      if (min_val == 0.0f && max_val == 0.0f) {
        backprops_wrt_input.chip<1>(i).device(d) = gradients_chip;
        auto min_chip = backprop_wrt_min.chip<0>(i);
        auto max_chip = backprop_wrt_max.chip<0>(i);
        min_chip.device(d) = min_chip.constant(0.0f);
        max_chip.device(d) = max_chip.constant(0.0f);
        continue;
      }
      float nudged_min, nudged_max, nudged_scale, inv_nudged_scale;
      Nudge(min_val, max_val, quant_min, quant_max, &nudged_min, &nudged_max,
            &nudged_scale, &inv_nudged_scale);

      const auto between_min_max =
          (inputs_chip >= nudged_min && inputs_chip <= nudged_max)
              .select(inputs_chip.constant(1.0f), inputs_chip.constant(0.0f));
      backprops_wrt_input.chip<1>(i).device(d) =
          gradients_chip * between_min_max;

      const auto below_min =
          (inputs_chip < nudged_min)
              .select(inputs_chip.constant(1.0f), inputs_chip.constant(0.0f));
      Eigen::DSizes<Index, 1> reduce(0);
      backprop_wrt_min.chip<0>(i).device(d) =
          (gradients_chip * below_min).sum(reduce);

      const auto above_max =
          (inputs_chip > nudged_max)
              .select(inputs_chip.constant(1.0f), inputs_chip.constant(0.0f));
      backprop_wrt_max.chip<0>(i).device(d) =
          (gradients_chip * above_max).sum(reduce);
    }
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_FAKE_QUANT_OPS_FUNCTOR_H_
