/* Copyright 2016 The TensorFlow Authors All Rights Reserved.

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

// TFProf representation of a Tensor's value.
// 1. Multi-dimension tensor is flattened in row major, and stored in proto.
// 2. integer are up-casted to int64. floats are up-casted to double. string
//    is not supported by TensorFlow CheckPointReader library, though it is
//    supported in current code.

#ifndef TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_TENSOR_H_
#define TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_TENSOR_H_
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
class MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_tensorDTh {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_tensorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_tensorDTh() {
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


#include <typeinfo>

#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/profiler/tfprof_output.pb.h"

namespace tensorflow {
namespace tfprof {

class TFProfTensor {
 public:
  explicit TFProfTensor(std::unique_ptr<Tensor> tensor)
      : tensor_(std::move(tensor)) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_tensorDTh mht_0(mht_0_v, 208, "", "./tensorflow/core/profiler/internal/tfprof_tensor.h", "TFProfTensor");

    Build();
  }

  // If pointers are provided, they are filled by the method.
  void Display(string* formatted_str, TFProfTensorProto* tfprof_tensor_pb);

 private:
  // Max length of tensor value displayed to CLI.
  const int64_t kTFProfTenosrMaxDisplayLen = 10000;
  // Max length after which a latency warning will be printed.
  const int64_t kTFProfTensorMaxWarnLen = 100000;

  void Build();

  template <typename T>
  bool AddValue(const T& value, TFProfTensorProto* dim) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_tensorDTh mht_1(mht_1_v, 227, "", "./tensorflow/core/profiler/internal/tfprof_tensor.h", "AddValue");

    std::ostringstream sstream;
    sstream << value;
    if (typeid(value) == typeid(double)) {
      double double_val = 0.0;
      CHECK(absl::SimpleAtod(sstream.str(), &double_val));  // Crash OK
      dim->add_value_double(double_val);
      absl::StrAppendFormat(&formatted_str_, "%.2f ",
                            dim->value_double(dim->value_double_size() - 1));
    } else if (typeid(value) == typeid(int64_t)) {
      int64_t int64_val = 0;
      CHECK(absl::SimpleAtoi(sstream.str(), &int64_val));  // Crash OK
      dim->add_value_int64(int64_val);
      absl::StrAppendFormat(&formatted_str_, "%d ",
                            dim->value_int64(dim->value_int64_size() - 1));
    } else if (typeid(value) == typeid(string)) {
      dim->add_value_str(sstream.str());
      absl::StrAppend(&formatted_str_, "'",
                      dim->value_str(dim->value_str_size() - 1), "' ");
    } else {
      CHECK(false) << "Unsupported type: " << typeid(value).name();
    }
  }

  // It assumes the flatten values are stored in row-major, which is mentioned
  // indirectly at various places:
  // TODO(xpan): Further verifying it.
  template <typename T>
  int64_t BuildOutput(int64_t start, int depth, const std::vector<T>& values,
                      TFProfTensorProto* dim) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_tensorDTh mht_2(mht_2_v, 259, "", "./tensorflow/core/profiler/internal/tfprof_tensor.h", "BuildOutput");

    formatted_str_ += "[";
    int64_t nstart = start;
    if (tensor_->dims() == 0 && values.size() == 1) {
      std::ostringstream sstream;
      sstream << values[nstart];

      if (typeid(values[nstart]) == typeid(double)) {
        double double_val = 0.0;
        CHECK(absl::SimpleAtod(sstream.str(), &double_val));  // Crash OK
        dim->add_value_double(double_val);
        absl::StrAppendFormat(&formatted_str_, "%.2f ",
                              dim->value_double(dim->value_double_size() - 1));
      } else if (typeid(values[nstart]) == typeid(int64_t)) {
        int64_t int64_val = 0;
        CHECK(absl::SimpleAtoi(sstream.str(), &int64_val));  // Crash OK
        dim->add_value_int64(int64_val);
        absl::StrAppendFormat(&formatted_str_, "%d ",
                              dim->value_int64(dim->value_int64_size() - 1));
      } else if (typeid(values[nstart]) == typeid(string)) {
        dim->add_value_str(sstream.str());
        absl::StrAppend(&formatted_str_, "'",
                        dim->value_str(dim->value_str_size() - 1), "' ");
      } else {
        CHECK(false) << "Unsupported type: " << typeid(values[nstart]).name();
      }
    } else {
      for (int i = 0; i < tensor_->dim_size(depth); i++) {
        // Last dimension, pull the values.
        if (depth == tensor_->dims() - 1) {
          std::ostringstream sstream;
          sstream << values[nstart];

          if (typeid(values[nstart]) == typeid(double)) {
            double double_val = 0.0;
            CHECK(absl::SimpleAtod(sstream.str(), &double_val));  // Crash OK
            dim->add_value_double(double_val);
            absl::StrAppendFormat(
                &formatted_str_, "%.2f ",
                dim->value_double(dim->value_double_size() - 1));
          } else if (typeid(values[nstart]) == typeid(int64_t)) {
            int64_t int64_val = 0;
            CHECK(absl::SimpleAtoi(sstream.str(), &int64_val));  // Crash OK
            dim->add_value_int64(int64_val);
            absl::StrAppendFormat(
                &formatted_str_, "%d ",
                dim->value_int64(dim->value_int64_size() - 1));
          } else if (typeid(values[nstart]) == typeid(string)) {
            dim->add_value_str(sstream.str());
            absl::StrAppend(&formatted_str_, "'",
                            dim->value_str(dim->value_str_size() - 1), "' ");
          } else {
            CHECK(false) << "Unsupported type: "
                         << typeid(values[nstart]).name();
          }
          ++nstart;
        } else {
          // Not-last dimension. Drill deeper.
          nstart = BuildOutput<T>(nstart, depth + 1, values, dim);
        }
      }
    }
    if (formatted_str_.length() > kTFProfTenosrMaxDisplayLen) {
      formatted_str_ = formatted_str_.substr(0, kTFProfTenosrMaxDisplayLen);
    }
    formatted_str_ += "],\n";
    return nstart;
  }

  template <typename T, typename U>
  void GetValueVec(std::vector<U>* value_vec) {
    // TODO(xpan): Address the huge tensor problem.
    if (tensor_->NumElements() > kTFProfTensorMaxWarnLen) {
      absl::FPrintF(stderr, "Showing huge tensor, the tool might halt...\n");
    }
    auto values = tensor_->flat<T>();
    for (int64_t i = 0; i < tensor_->NumElements(); i++) {
      value_vec->push_back(static_cast<U>(values(i)));
    }
  }

  TFProfTensorProto tfprof_tensor_pb_;
  std::unique_ptr<Tensor> tensor_;
  string formatted_str_;
};
}  // namespace tfprof
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_TENSOR_H_
