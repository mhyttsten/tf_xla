/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_UTIL_MATMUL_BCAST_H_
#define TENSORFLOW_CORE_UTIL_MATMUL_BCAST_H_
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
class MHTracer_DTPStensorflowPScorePSutilPSmatmul_bcastDTh {
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
   MHTracer_DTPStensorflowPScorePSutilPSmatmul_bcastDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSmatmul_bcastDTh() {
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


#include <vector>

#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/util/bcast.h"

namespace tensorflow {

// Simple wrapper over BCast specialized for MatMul.
// Provides utilities for broadcasting across batch dimensions for binary
// MatMul-like operations. If neither argument has batch dimensions (rank <= 2)
// then no broadcasting is needed and the operation MatMul operation is
// considered valid.
class MatMulBCast {
 public:
  using Vec = BCast::Vec;

  MatMulBCast(const Vec& x, const Vec& y) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSmatmul_bcastDTh mht_0(mht_0_v, 205, "", "./tensorflow/core/util/matmul_bcast.h", "MatMulBCast");

    if (std::max(x.size(), y.size()) == 2) return;
    const Vec x_resized(x.begin(), x.end() - 2);
    const Vec y_resized(y.begin(), y.end() - 2);

    batch_bcast_ =
        absl::make_unique<BCast>(std::move(x_resized), std::move(y_resized));
    if (!batch_bcast_->IsValid()) {
      // Set broadcasting_required_ to true to make IsValid() return false;
      broadcasting_required_ = true;
      return;
    }

    x_batch_size_ = TensorShape(batch_bcast_->x_reshape()).num_elements();
    y_batch_size_ = TensorShape(batch_bcast_->y_reshape()).num_elements();
    output_batch_shape_ = TensorShape(batch_bcast_->output_shape());
    output_batch_size_ = output_batch_shape_.num_elements();
    broadcasting_required_ =
        std::min(x_batch_size_, y_batch_size_) != output_batch_size_;

    if (broadcasting_required_) {
      ComputeBatchIndices(output_batch_size_, batch_bcast_->x_reshape(),
                          batch_bcast_->x_bcast(), &x_batch_indices_);
      ComputeBatchIndices(output_batch_size_, batch_bcast_->y_reshape(),
                          batch_bcast_->y_bcast(), &y_batch_indices_);
    }
  }

  bool IsValid() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSmatmul_bcastDTh mht_1(mht_1_v, 236, "", "./tensorflow/core/util/matmul_bcast.h", "IsValid");

    return !broadcasting_required_ || (batch_bcast_ && batch_bcast_->IsValid());
  }
  bool IsBroadcastingRequired() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPSmatmul_bcastDTh mht_2(mht_2_v, 242, "", "./tensorflow/core/util/matmul_bcast.h", "IsBroadcastingRequired");
 return broadcasting_required_; }

  const int64_t output_batch_size() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPSmatmul_bcastDTh mht_3(mht_3_v, 247, "", "./tensorflow/core/util/matmul_bcast.h", "output_batch_size");
 return output_batch_size_; }
  const int64_t x_batch_size() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSutilPSmatmul_bcastDTh mht_4(mht_4_v, 251, "", "./tensorflow/core/util/matmul_bcast.h", "x_batch_size");
 return x_batch_size_; }
  const int64_t y_batch_size() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSutilPSmatmul_bcastDTh mht_5(mht_5_v, 255, "", "./tensorflow/core/util/matmul_bcast.h", "y_batch_size");
 return y_batch_size_; }
  const TensorShape& output_batch_shape() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSutilPSmatmul_bcastDTh mht_6(mht_6_v, 259, "", "./tensorflow/core/util/matmul_bcast.h", "output_batch_shape");
 return output_batch_shape_; }

  // Returns the mapping from the flattened output batch indices to x's
  // flattened batch indices. The result is a vector of length
  // output_batch_size(). To compute the i'th batch output, a binary matmul-like
  // operation should use the `x_batch_indices()[i]`th batch index of `x`.
  // Note: Returns an empty vector if broadcasting is not required. Callers
  // should only use this when IsBroadcastingRequired() returns true.
  const std::vector<int64_t>& x_batch_indices() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSutilPSmatmul_bcastDTh mht_7(mht_7_v, 270, "", "./tensorflow/core/util/matmul_bcast.h", "x_batch_indices");

    return x_batch_indices_;
  }
  // Returns the mapping from the flattened output batch indices to y's
  // flattened batch indices. Similar to x_batch_indices().
  // Note: Returns an empty vector if broadcasting is not required. Callers
  // should only use this when IsBroadcastingRequired() returns true.
  const std::vector<int64_t>& y_batch_indices() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSutilPSmatmul_bcastDTh mht_8(mht_8_v, 280, "", "./tensorflow/core/util/matmul_bcast.h", "y_batch_indices");

    return y_batch_indices_;
  }

 private:
  std::unique_ptr<BCast> batch_bcast_;
  bool broadcasting_required_ = false;
  int64_t x_batch_size_ = 1;
  int64_t y_batch_size_ = 1;
  TensorShape output_batch_shape_;
  int64_t output_batch_size_ = 1;
  std::vector<int64_t> x_batch_indices_;
  std::vector<int64_t> y_batch_indices_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_MATMUL_BCAST_H_
