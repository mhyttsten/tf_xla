/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_TRT_TENSOR_PROXY_H
#define TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_TRT_TENSOR_PROXY_H
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
class MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh {
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
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh() {
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


#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/tf2tensorrt/common/utils.h"
#include "tensorflow/core/platform/logging.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "third_party/tensorrt/NvInfer.h"

namespace tensorflow {

namespace tensorrt {

// SimpleITensor implements part of the ITensor interfaces to support the TF-TRT
// validator, as well as some TF-TRT tests. The former use case only utilizes
// the interfaces related to shape and type information.
class SimpleITensor {
 public:
  SimpleITensor(nvinfer1::DataType trt_dtype, const nvinfer1::Dims& trt_dims)
      : trt_dtype_(trt_dtype), trt_dims_(trt_dims) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_0(mht_0_v, 209, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "SimpleITensor");
}

  SimpleITensor() : dynamic_range_min_(0.0f), dynamic_range_max_(0.0f) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_1(mht_1_v, 214, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "SimpleITensor");
}
  SimpleITensor(const nvinfer1::Dims& dims)
      : trt_dims_(dims), dynamic_range_min_(0.0f), dynamic_range_max_(0.0f) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_2(mht_2_v, 219, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "SimpleITensor");
}

  SimpleITensor(const std::vector<int>& dims) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_3(mht_3_v, 224, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "SimpleITensor");

    trt_dims_.nbDims = dims.size();
    for (int i = 0; i < dims.size(); ++i) {
      trt_dims_.d[i] = dims[i];
    }
    dynamic_range_min_ = 0.0f;
    dynamic_range_max_ = 0.0f;
  }

  void setName(const char* name) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_4(mht_4_v, 237, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "setName");
}

  const char* getName() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_5(mht_5_v, 242, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "getName");
 return ""; }

  void setDimensions(nvinfer1::Dims dimensions) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_6(mht_6_v, 247, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "setDimensions");
 trt_dims_ = dimensions; }

  nvinfer1::Dims getDimensions() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_7(mht_7_v, 252, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "getDimensions");
 return trt_dims_; }

  void setType(nvinfer1::DataType trt_dtype) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_8(mht_8_v, 257, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "setType");
 trt_dtype_ = trt_dtype; }

  nvinfer1::DataType getType() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_9(mht_9_v, 262, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "getType");
 return trt_dtype_; }

  bool isNetworkInput() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_10(mht_10_v, 267, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "isNetworkInput");
 return false; }

  bool isNetworkOutput() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_11(mht_11_v, 272, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "isNetworkOutput");
 return false; }

  void setBroadcastAcrossBatch(bool broadcastAcrossBatch) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_12(mht_12_v, 277, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "setBroadcastAcrossBatch");
}

  bool getBroadcastAcrossBatch() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_13(mht_13_v, 282, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "getBroadcastAcrossBatch");
 return false; }

  nvinfer1::TensorLocation getLocation() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_14(mht_14_v, 287, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "getLocation");
 return location_; }

  void setLocation(nvinfer1::TensorLocation location) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_15(mht_15_v, 292, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "setLocation");
 location_ = location; }
  bool setDynamicRange(float min, float max) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_16(mht_16_v, 296, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "setDynamicRange");

    dynamic_range_max_ = max;
    dynamic_range_min_ = min;
    return true;
  }

  float getDynamicRange() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_17(mht_17_v, 305, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "getDynamicRange");

    return (std::abs(dynamic_range_min_) + dynamic_range_max_) / 2.f;
  }
  bool dynamicRangeIsSet() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_18(mht_18_v, 311, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "dynamicRangeIsSet");
 return true; }

  void resetDynamicRange() {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_19(mht_19_v, 316, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "resetDynamicRange");

    dynamic_range_min_ = 0.f;
    dynamic_range_max_ = 0.f;
  }
  float getDynamicRangeMin() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_20(mht_20_v, 323, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "getDynamicRangeMin");
 return dynamic_range_min_; }

  float getDynamicRangeMax() const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_21(mht_21_v, 328, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "getDynamicRangeMax");
 return dynamic_range_max_; }

  void setAllowedFormats(nvinfer1::TensorFormats formats) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_22(mht_22_v, 333, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "setAllowedFormats");
}

  nvinfer1::TensorFormats getAllowedFormats() const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_23(mht_23_v, 338, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "getAllowedFormats");
 return 1; }

  bool isShapeTensor() const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_24(mht_24_v, 343, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "isShapeTensor");
 return false; }
  bool isExecutionTensor() const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_25(mht_25_v, 347, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "isExecutionTensor");
 return true; }

 private:
  nvinfer1::DataType trt_dtype_;
  nvinfer1::Dims trt_dims_;
  std::string name_;
  nvinfer1::TensorLocation location_;
  float dynamic_range_min_;
  float dynamic_range_max_;
};

enum class TensorType : int { kTRT, kSIMPLE };

class ITensorProxy {
 public:
  //! ITensor not owned
  ITensorProxy(nvinfer1::ITensor* trt_tensor)
      : trt_tensor_(trt_tensor), ttype_(TensorType::kTRT) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_26(mht_26_v, 367, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "ITensorProxy");
}

  //! SimpleITensor owned
  ITensorProxy(SimpleITensor* simple_itensor)
      : simple_tensor_(simple_itensor), ttype_(TensorType::kSIMPLE) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_27(mht_27_v, 374, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "ITensorProxy");
}

  //! SimpleITensor owned
  explicit ITensorProxy(nvinfer1::DataType trt_dtype,
                        const nvinfer1::Dims& trt_dims)
      : simple_tensor_(std::unique_ptr<SimpleITensor>(
            new SimpleITensor(trt_dtype, trt_dims))),
        ttype_(TensorType::kSIMPLE) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_28(mht_28_v, 384, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "ITensorProxy");
}

  //! Variants for testing purposes
  ITensorProxy()
      : simple_tensor_(std::unique_ptr<SimpleITensor>(new SimpleITensor())),
        ttype_(TensorType::kSIMPLE) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_29(mht_29_v, 392, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "ITensorProxy");
}

  explicit ITensorProxy(const nvinfer1::Dims& dims)
      : simple_tensor_(std::unique_ptr<SimpleITensor>(new SimpleITensor(dims))),
        ttype_(TensorType::kSIMPLE) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_30(mht_30_v, 399, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "ITensorProxy");
}

  explicit ITensorProxy(const std::vector<int>& dims)
      : simple_tensor_(std::unique_ptr<SimpleITensor>(new SimpleITensor(dims))),
        ttype_(TensorType::kSIMPLE) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_31(mht_31_v, 406, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "ITensorProxy");
}

  bool is_trt_tensor() const {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_32(mht_32_v, 411, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "is_trt_tensor");

    CHECK(validate());
    return trt_tensor_ != nullptr;
  }

  bool is_simple_tensor() const {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_33(mht_33_v, 419, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "is_simple_tensor");

    CHECK(validate());
    return simple_tensor_ != nullptr;
  }

  TensorType ttype() const {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_34(mht_34_v, 427, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "ttype");
 return ttype_; }

  nvinfer1::ITensor* trt_tensor() const {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_35(mht_35_v, 432, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "trt_tensor");

    CHECK_NOTNULL(trt_tensor_);
    CHECK(ttype_ == TensorType::kTRT);
    return trt_tensor_;
  }

  SimpleITensor* simple_tensor() const {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_36(mht_36_v, 441, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "simple_tensor");

    CHECK_NOTNULL(simple_tensor_);
    CHECK(ttype_ == TensorType::kSIMPLE);
    return simple_tensor_.get();
  }

  void setName(const char* name) {
   std::vector<std::string> mht_37_v;
   mht_37_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_37(mht_37_v, 451, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "setName");

    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->setName(name);
      case TensorType::kSIMPLE:
        return simple_tensor_->setName(name);
    }
    LOG(FATAL) << "Unsupported itensor_ type";
  }

  const char* getName() const {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_38(mht_38_v, 464, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "getName");

    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->getName();
      case TensorType::kSIMPLE:
        return simple_tensor_->getName();
    }
    LOG(FATAL) << "Unsupported itensor_ type";
  }

  void setDimensions(nvinfer1::Dims dimensions) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_39(mht_39_v, 477, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "setDimensions");

    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->setDimensions(dimensions);
      case TensorType::kSIMPLE:
        return simple_tensor_->setDimensions(dimensions);
    }
    LOG(FATAL) << "Unsupported itensor_ type";
  }

  nvinfer1::Dims getDimensions() const {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_40(mht_40_v, 490, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "getDimensions");

    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->getDimensions();
      case TensorType::kSIMPLE:
        return simple_tensor_->getDimensions();
    }
    LOG(FATAL) << "Unsupported itensor_ type";
  }

  void setType(nvinfer1::DataType type) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_41(mht_41_v, 503, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "setType");

    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->setType(type);
      case TensorType::kSIMPLE:
        return simple_tensor_->setType(type);
    }
    LOG(FATAL) << "Unsupported itensor_ type";
  }

  nvinfer1::DataType getType() const {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_42(mht_42_v, 516, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "getType");

    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->getType();
      case TensorType::kSIMPLE:
        return simple_tensor_->getType();
    }
    LOG(FATAL) << "Unsupported itensor_ type";
  }

  bool isNetworkInput() const {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_43(mht_43_v, 529, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "isNetworkInput");

    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->isNetworkInput();
      case TensorType::kSIMPLE:
        return simple_tensor_->isNetworkInput();
    }
    LOG(FATAL) << "Unsupported itensor_ type";
  }

  bool isNetworkOutput() const {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_44(mht_44_v, 542, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "isNetworkOutput");

    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->isNetworkOutput();
      case TensorType::kSIMPLE:
        return simple_tensor_->isNetworkOutput();
    }
    LOG(FATAL) << "Unsupported itensor_ type";
  }

  void setBroadcastAcrossBatch(bool broadcastAcrossBatch) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_45(mht_45_v, 555, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "setBroadcastAcrossBatch");

    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->setBroadcastAcrossBatch(broadcastAcrossBatch);
      case TensorType::kSIMPLE:
        return simple_tensor_->setBroadcastAcrossBatch(broadcastAcrossBatch);
    }
    LOG(FATAL) << "Unsupported itensor_ type";
  }

  bool getBroadcastAcrossBatch() const {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_46(mht_46_v, 568, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "getBroadcastAcrossBatch");

    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->getBroadcastAcrossBatch();
      case TensorType::kSIMPLE:
        return simple_tensor_->getBroadcastAcrossBatch();
    }
    LOG(FATAL) << "Unsupported itensor_ type";
  }

  nvinfer1::TensorLocation getLocation() const {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_47(mht_47_v, 581, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "getLocation");

    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->getLocation();
      case TensorType::kSIMPLE:
        return simple_tensor_->getLocation();
    }
    LOG(FATAL) << "Unsupported itensor_ type";
  }

  void setLocation(nvinfer1::TensorLocation location) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_48(mht_48_v, 594, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "setLocation");

    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->setLocation(location);
      case TensorType::kSIMPLE:
        return simple_tensor_->setLocation(location);
    }
    LOG(FATAL) << "Unsupported itensor_ type";
  }

  bool setDynamicRange(float min, float max) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_49(mht_49_v, 607, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "setDynamicRange");

    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->setDynamicRange(min, max);
      case TensorType::kSIMPLE:
        return simple_tensor_->setDynamicRange(min, max);
    }
    LOG(FATAL) << "Unsupported itensor_ type";
  }

  bool dynamicRangeIsSet() const {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_50(mht_50_v, 620, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "dynamicRangeIsSet");

    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->dynamicRangeIsSet();
      case TensorType::kSIMPLE:
        return simple_tensor_->dynamicRangeIsSet();
    }
    LOG(FATAL) << "Unsupported itensor_ type";
  }

  void resetDynamicRange() {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_51(mht_51_v, 633, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "resetDynamicRange");

    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->resetDynamicRange();
      case TensorType::kSIMPLE:
        return simple_tensor_->resetDynamicRange();
    }
    LOG(FATAL) << "Unsupported itensor_ type";
  }
  float getDynamicRangeMin() const {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_52(mht_52_v, 645, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "getDynamicRangeMin");

    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->getDynamicRangeMin();
      case TensorType::kSIMPLE:
        return simple_tensor_->getDynamicRangeMin();
    }
    LOG(FATAL) << "Unsupported itensor_ type";
  }

  float getDynamicRangeMax() const {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_53(mht_53_v, 658, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "getDynamicRangeMax");

    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->getDynamicRangeMax();
      case TensorType::kSIMPLE:
        return simple_tensor_->getDynamicRangeMax();
    }
    LOG(FATAL) << "Unsupported itensor_ type";
  }
#if !IS_TRT_VERSION_GE(8, 0, 0, 0)
  float getDynamicRange() const {
    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->getDynamicRange();
      case TensorType::kSIMPLE:
        return simple_tensor_->getDynamicRange();
    }
    LOG(FATAL) << "Unsupported itensor_ type";
  }
#endif
  void setAllowedFormats(nvinfer1::TensorFormats formats) {
    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->setAllowedFormats(formats);
      case TensorType::kSIMPLE:
        return simple_tensor_->setAllowedFormats(formats);
    }
    LOG(FATAL) << "Unsupported itensor_ type";
  }

  nvinfer1::TensorFormats getAllowedFormats() const {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_54(mht_54_v, 691, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "getAllowedFormats");

    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->getAllowedFormats();
      case TensorType::kSIMPLE:
        return simple_tensor_->getAllowedFormats();
    }
    LOG(FATAL) << "Unsupported itensor_ type";
  }

  bool isShapeTensor() const {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_55(mht_55_v, 704, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "isShapeTensor");

    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->isShapeTensor();
      case TensorType::kSIMPLE:
        return simple_tensor_->isShapeTensor();
    }
    LOG(FATAL) << "Unsupported itensor_ type";
  }

  bool isExecutionTensor() const {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_56(mht_56_v, 717, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "isExecutionTensor");

    switch (ttype_) {
      case TensorType::kTRT:
        return trt_tensor_->isExecutionTensor();
      case TensorType::kSIMPLE:
        return simple_tensor_->isExecutionTensor();
    }
    LOG(FATAL) << "Unsupported itensor_ type";
  }

 private:
  bool validate() const {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_57(mht_57_v, 731, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "validate");

    return (trt_tensor_ && !simple_tensor_) || (!trt_tensor_ && simple_tensor_);
  }

  // When ITensorProxy represents an ITensor, the ITensor can be either passed
  // by the caller via the constructor that takes an ITensor* as parameter, or
  // be created as a SimpleITensor.
  //
  // In the first case, the ITensor pointer is stored in 'tensor_' below, and
  // the ITensor itself is not owned by this class. This method is used by
  // Converter (e.g. AddInputTensor) and op converters during TRT network
  // construction, where the TRT network owns the ITensor.
  //
  nvinfer1::ITensor* trt_tensor_ = nullptr;  // Not owned.
  // In the second case, the created SimpleITensor is stored in
  // 'simple_itensor_' below and is owned by this class. SimpleITensor is a fake
  // implementation of ITensor and is used for testing and by TrtNodeValidator
  //  to validate the graph nodes.
  std::shared_ptr<SimpleITensor> simple_tensor_ = nullptr;

  TensorType ttype_;
};

class ITensorProxyPtr {
 public:
  ITensorProxyPtr(std::nullptr_t) : p_(nullptr) {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_58(mht_58_v, 759, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "ITensorProxyPtr");
}
  ITensorProxyPtr(ITensorProxy* p) : p_(p) {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_59(mht_59_v, 763, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "ITensorProxyPtr");
}
  ITensorProxyPtr(nvinfer1::ITensor* p) : p_(new ITensorProxy(p)) {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_60(mht_60_v, 767, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "ITensorProxyPtr");
}
  ITensorProxyPtr(SimpleITensor* p) : p_(new ITensorProxy(p)) {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_61(mht_61_v, 771, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "ITensorProxyPtr");
}

  ITensorProxyPtr() : p_(new ITensorProxy()) {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_62(mht_62_v, 776, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "ITensorProxyPtr");
}
  ITensorProxyPtr(const nvinfer1::Dims& dims) : p_(new ITensorProxy(dims)) {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_63(mht_63_v, 780, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "ITensorProxyPtr");
}
  ITensorProxyPtr(const std::vector<int>& dims) : p_(new ITensorProxy(dims)) {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_64(mht_64_v, 784, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "ITensorProxyPtr");
}

  std::shared_ptr<ITensorProxy> p_{nullptr};
  ITensorProxy* operator->() { return p_.get(); }
  ITensorProxy* operator->() const { return p_.get(); }
  ITensorProxy* operator*() {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_65(mht_65_v, 792, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "*");
 return p_.get(); }
  ITensorProxy* operator*() const {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_tensor_proxyDTh mht_66(mht_66_v, 796, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h", "*");
 return p_.get(); }
};

inline bool operator==(const ITensorProxyPtr& p1, const ITensorProxyPtr& p2) {
  if (p1.p_ == nullptr) {
    return p2.p_ == nullptr;
  }
  if (p2.p_ == nullptr) {
    return p1.p_ == nullptr;
  }
  return (p1->ttype() == p2->ttype()) &&
         ((p1->ttype() == TensorType::kTRT &&
           p1->trt_tensor() == p2->trt_tensor()) ||
          (p1->ttype() == TensorType::kSIMPLE &&
           p1->simple_tensor() == p2->simple_tensor()));
}

inline bool operator!=(const ITensorProxyPtr& p1, const ITensorProxyPtr& p2) {
  return !(p1 == p2);
}

struct ITensorProxyHash {
  size_t operator()(const ITensorProxyPtr& tensor) const {
    return reinterpret_cast<std::uintptr_t>(tensor.p_.get());
  }
};

}  // namespace tensorrt
}  // namespace tensorflow
#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT

#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_TRT_TENSOR_PROXY_H
