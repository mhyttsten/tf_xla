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

#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_UTILS_TRT_SHAPE_OPTIMIZATION_PROFILES_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_UTILS_TRT_SHAPE_OPTIMIZATION_PROFILES_H_
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
class MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_shape_optimization_profilesDTh {
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
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_shape_optimization_profilesDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_shape_optimization_profilesDTh() {
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


#include <list>
#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow/compiler/tf2tensorrt/common/datavec.h"
#include "tensorflow/compiler/tf2tensorrt/convert/trt_parameters.h"
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_execution_context.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_logger.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include "third_party/tensorrt/NvInfer.h"

namespace tensorflow {
namespace tensorrt {

// Stores optimization profile parameters (min/opt/max of each input shape).
//
// A TensorRT optimization profile describes the possible min/max values of
// each dynamic input shape along with an optimum value. These values are used
// by the TensorRT builder to select the best kernel for the optimum value among
// those kernels that are valid for all input tensors in the [min, max] range.
struct OptimizationProfileConfig {
  // Length of vector == 2*num_inputs to engine. min[0:num_inputs-1] are the min
  // input dimensions for execution tensors. If engine has shape input tensors,
  // then min[num_inputs + i] store the shape value for input i. For inputs that
  // are not shape tensors min = opt = max = {0, {}}.
  //
  // When the OptimizationProfileConfig is created from the network definition
  // (AddProfiles), then each elements of the min, opt, max vectors are defined.
  // When the OptimizationProfileConfig object is restored during engine
  // deserialization (RestoreProfiles), then some inputs can be pruned
  // (see TrtShapeOptimizationProfile::is_pruned_input_). In that case min[i]
  // is not defined for pruned inputs (same is true for opt and max).
  std::vector<nvinfer1::Dims> min;
  std::vector<nvinfer1::Dims> opt;
  std::vector<nvinfer1::Dims> max;

  string DebugString() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_shape_optimization_profilesDTh mht_0(mht_0_v, 234, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_shape_optimization_profiles.h", "DebugString");

    using absl::StrCat;
    return StrCat("[min: ", tensorflow::tensorrt::DebugString(min),
                  ", opt: : ", tensorflow::tensorrt::DebugString(opt),
                  ", max: ", tensorflow::tensorrt::DebugString(max), "]");
  }

  // Sets the min/opt/max dimensions for profile.
  //
  // The given min/opt/max dimensions should satisfy the condition
  // min <= opt <= max. Additionally TRT requires that the min/opt/max values
  // are compatible with the network input. Compatibility is defined the
  // following way: let dim be the shape of an input binding and min/opt/max the
  // corresponding profile dims. TRT requires that dim.d[k] must be -1 if
  // (min.d[k] != dim.d[k] || opt.d[k] != dim.d[k] || max.d[k] != dim.d[k]).
  //
  // Parameters:
  // network - TensorRT network, used to enumerate all the input tensors
  // profile - on exit the profile information will be set for each input tensor
  Status SetDimensions(const nvinfer1::INetworkDefinition* network,
                       nvinfer1::IOptimizationProfile* profile) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_shape_optimization_profilesDTh mht_1(mht_1_v, 257, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_shape_optimization_profiles.h", "SetDimensions");

    int n_inputs = network->getNbInputs();
    if (min.size() != 2 * n_inputs || opt.size() != 2 * n_inputs ||
        max.size() != 2 * n_inputs) {
      return errors::Internal("Incorrect number of profile config parameters");
    }
    for (int i = 0; i < n_inputs; i++) {
      const ITensorProxyPtr input = network->getInput(i);
      const char* name = input->getName();
      if (input->isShapeTensor()) {
        int idx = i + n_inputs;
        VLOG(2) << "Setting shape values for " << name << ", "
                << ::tensorflow::tensorrt::DebugString(opt[idx]);
        profile->setShapeValues(name, nvinfer1::OptProfileSelector::kMIN,
                                min[idx].d, min[idx].nbDims);
        profile->setShapeValues(name, nvinfer1::OptProfileSelector::kOPT,
                                opt[idx].d, opt[idx].nbDims);
        profile->setShapeValues(name, nvinfer1::OptProfileSelector::kMAX,
                                max[idx].d, max[idx].nbDims);
      }
      VLOG(2) << "Setting input dimensions for " << name << ", "
              << ::tensorflow::tensorrt::DebugString(opt[i]);
      profile->setDimensions(name, nvinfer1::OptProfileSelector::kMIN, min[i]);
      profile->setDimensions(name, nvinfer1::OptProfileSelector::kOPT, opt[i]);
      profile->setDimensions(name, nvinfer1::OptProfileSelector::kMAX, max[i]);
    }
    return Status::OK();
  }

  // Returns true if profile range completely includes the given shapes.
  bool IncludesShapes(const std::vector<TensorShape>& shapes,
                      bool has_shape_tensor,
                      const std::vector<nvinfer1::Dims>& shape_values,
                      const std::vector<bool>& is_pruned_input) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_shape_optimization_profilesDTh mht_2(mht_2_v, 293, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_shape_optimization_profiles.h", "IncludesShapes");

    // min, max, and opt must have the same size which is already verified in
    // SetDimensions.
    if (min.size() != shapes.size() * 2 ||
        (has_shape_tensor && min.size() != shape_values.size() * 2)) {
      VLOG(2) << "Profile size mismatch min size " << min.size()
              << " vs input shapes size " << shapes.size() << " "
              << shape_values.size();
      return false;
    }
    for (int i = 0; i < shapes.size(); i++) {
      if (is_pruned_input[i]) {
        continue;
      }
      auto current_shape = shapes[i];
      // min, max, and opt must have the same nbDims, which is already verified
      // in SetDimensions.
      if (min[i].nbDims != current_shape.dims()) {
        return false;
      }
      // Check if range [min, max] includes current_shape.
      for (int dim = 0; dim < current_shape.dims(); dim++) {
        if ((min[i].d[dim] > current_shape.dim_size(dim)) ||
            (max[i].d[dim] < current_shape.dim_size(dim))) {
          return false;
        }
      }
    }
    // Check shape values.
    if (has_shape_tensor) {
      int offset = shapes.size();
      for (int i = 0; i < shape_values.size(); i++) {
        if (is_pruned_input[i]) {
          continue;
        }
        auto shape_val = shape_values[i];
        // min, max, and opt must have the same nbDims, which is already
        // verified in SetDimensions.
        if (min[i + offset].nbDims != shape_val.nbDims) {
          return false;
        }
        // Check if range [min, max] includes shape_val.
        for (int dim = 0; dim < shape_val.nbDims; dim++) {
          if (min[i + offset].d[dim] > shape_val.d[dim] ||
              max[i + offset].d[dim] < shape_val.d[dim]) {
            return false;
          }
        }
      }
    }
    return true;
  }
};

// Manages Optimization profiles during TRT Engine construction.
//
// An optimization profile describes a range of dimensions for each TRT network
// input, and the optimal dimensions that the auto-tuner should use for
// optimization.
//
// This class stores the list of input shapes that were seen during the
// build/profile_generation_mode phase, and using them it creates a set of
// OptimizationProfileConfigs. These configs will be added to IBuilderConfig
// before the engine is created.
class TrtShapeOptimizationProfile {
 public:
  TrtShapeOptimizationProfile() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_shape_optimization_profilesDTh mht_3(mht_3_v, 362, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_shape_optimization_profiles.h", "TrtShapeOptimizationProfile");
}

  // Stores input shape information during profile_generation_mode.
  void AddShape(const std::vector<TensorShape>& shapes) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_shape_optimization_profilesDTh mht_4(mht_4_v, 368, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_shape_optimization_profiles.h", "AddShape");

    input_shapes_.push_back(shapes);
    input_shape_values_.push_back(actual_shape_values_);
    VLOG(1) << "Collected shape(s) " << DebugString(shapes) << " for profiles.";
  }

  // Collects ShapeTensorCompatible tensor values. This is needed both during
  // profile_generation_mode and during normal inference calls.
  Status CollectShapeValues(OpKernelContext* ctx);

  // Collects ShapeTensorCompatible tensor values, used only for unit tests.
  Status CollectShapeValues(const DataVec& input);

  void clear() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_shape_optimization_profilesDTh mht_5(mht_5_v, 384, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_shape_optimization_profiles.h", "clear");
 profiles_.clear(); }

  // Returns the profile number that should be used to execute the network with
  // the given input shapes. Returns -1 if none of cached profiles are
  // compatible with the given input shapes.
  int GetProfileNumber(const std::vector<TensorShape>& shapes);

  // Creates optimization profiles and add them to the builder config.
  Status ConfigureBuilder(nvinfer1::IBuilder* builder,
                          nvinfer1::IBuilderConfig* config,
                          const nvinfer1::INetworkDefinition* network);

  // Creates execution contexts for each optimization profile.
  Status CreateExecutionContexts(nvinfer1::ICudaEngine* engine,
                                 std::vector<ExecutionContext>* exec_contexts);

  Status SetInputShapeBinding(int input_index, int binding_index,
                              nvinfer1::ICudaEngine* cuda_engine,
                              nvinfer1::IExecutionContext* exec_context) const;

  // Creates optimization profiles profiles_ for the set of concrete input
  // shapes collected in input_shapes_. The input_partial_shapes of the network
  // is used to ensure that the created optimization profiles are compatible
  // with the network.
  void InitProfiles(const std::vector<PartialTensorShape>& input_partial_shapes,
                    ProfileStrategy strategy);

  // Returns number of created profiles.
  int GetNumProfiles() const;

  bool HasShape() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_shape_optimization_profilesDTh mht_6(mht_6_v, 417, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_shape_optimization_profiles.h", "HasShape");
 return !input_shapes_.empty(); }
  bool NeedProfiles() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_shape_optimization_profilesDTh mht_7(mht_7_v, 421, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_shape_optimization_profiles.h", "NeedProfiles");
 return need_profiles_; }

  // Restores profiles from the engine (used after deserialization).
  Status RestoreProfiles(const nvinfer1::ICudaEngine* engine,
                         int n_network_inputs);

  // Whether the network has any shape tensors.
  bool HasShapeTensor() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_shape_optimization_profilesDTh mht_8(mht_8_v, 431, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_shape_optimization_profiles.h", "HasShapeTensor");
 return has_shape_tensor_; }

  void SetShapeTensorMask(const nvinfer1::INetworkDefinition* network);

  // Whether the optimization profiles describe input that can be handled with
  // a static engine (only 1 profile with min=max).
  bool IsStaticCompatible() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_shape_optimization_profilesDTh mht_9(mht_9_v, 440, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_shape_optimization_profiles.h", "IsStaticCompatible");

    return strategy_ == ProfileStrategy::kOptimal && profiles_.size() == 1 &&
           !HasShapeTensor();
    // TODO(tfeher): remove !HasShapeTensor() condition once the
    // FixShapeValueProfile workaround is turned off.
  }

 private:
  // Set of input shape vetors that we collect during profile_generation_mode.
  std::vector<std::vector<TensorShape>> input_shapes_;

  // Input shape values that we collect during profile_generation_mode. If the
  // tensor is not compatible with a TRT shape tensor then an empty shape is
  // stored.
  std::vector<std::vector<nvinfer1::Dims>> input_shape_values_;

  // Shape values present in the current inference call.
  std::vector<nvinfer1::Dims> actual_shape_values_;

  // The optimization profiles generated from input_shapes_.
  std::vector<OptimizationProfileConfig> profiles_;

  // Whether the network has any shape tensors. Initially we assume that the
  // network might have a shape value input. This will be updated when the
  // network is created / engine is deserialized.
  bool has_shape_tensor_ = true;

  // Whether the network/engine requires optimization profiles.
  bool need_profiles_ = false;

  // Whether an input tensor is a shape tensor.
  std::vector<bool> is_shape_tensor_;

  // Whether a network input was pruned (only in TRT 7).
  std::vector<bool> is_pruned_input_;

  // Optimization profile generation strategy.
  ProfileStrategy strategy_;

  // Adds optimization profiles to the builder config.
  Status AddProfiles(nvinfer1::IBuilder* builder,
                     nvinfer1::IBuilderConfig* config,
                     const nvinfer1::INetworkDefinition* network);

  void SetShapeTensorMask(const nvinfer1::ICudaEngine* engine, int n_inputs);
  void SetShapeTensorMask(
      const std::vector<PartialTensorShape>& input_partial_shapes);

  Status SetPrunedMask(const nvinfer1::ICudaEngine* engine,
                       int n_network_inputs);

  void ImplicitBatchModeCompatibleStrategy(
      const std::vector<std::vector<nvinfer1::Dims>>& collected_shapes);
  void OptimalStrategy(
      const std::vector<std::vector<nvinfer1::Dims>>& collected_shapes);
  Status RangeStrategy(
      const std::vector<std::vector<nvinfer1::Dims>>& collected_shapes);
};

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_UTILS_TRT_SHAPE_OPTIMIZATION_PROFILES_H_
