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

#ifndef TENSORFLOW_CORE_KERNELS_POISSON_LOSS_H_
#define TENSORFLOW_CORE_KERNELS_POISSON_LOSS_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSpoissonSLlossDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSpoissonSLlossDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSpoissonSLlossDTh() {
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


#include <cmath>

#include "tensorflow/core/kernels/loss.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

class PoissonLossUpdater : public DualLossUpdater {
 public:
  // Update is found by a Newton algorithm (see readme.md).
  double ComputeUpdatedDual(const int num_loss_partitions, const double label,
                            const double example_weight,
                            const double current_dual, const double wx,
                            const double weighted_example_norm) const final {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSpoissonSLlossDTh mht_0(mht_0_v, 201, "", "./tensorflow/core/kernels/poisson-loss.h", "ComputeUpdatedDual");

    // Newton algorithm converges quadratically so 10 steps will be largely
    // enough to achieve a very good precision
    static const int newton_total_steps = 10;
    // Initialize the Newton optimization at x such that
    // exp(x) = label - current_dual
    const double y_minus_a = label - current_dual;
    double x = (y_minus_a > 0) ? log(y_minus_a) : 0;
    for (int i = 0; i < newton_total_steps; ++i) {
      x = NewtonStep(x, num_loss_partitions, label, wx, example_weight,
                     weighted_example_norm, current_dual);
    }
    return label - exp(x);
  }

  // Dual of poisson loss function.
  // https://en.wikipedia.org/wiki/Convex_conjugate
  double ComputeDualLoss(const double current_dual, const double example_label,
                         const double example_weight) const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSpoissonSLlossDTh mht_1(mht_1_v, 222, "", "./tensorflow/core/kernels/poisson-loss.h", "ComputeDualLoss");

    // Dual of the poisson loss function is
    // (y-a)*(log(y-a)-1), where a is the dual variable.
    // It is defined only for a<y.
    const double y_minus_a = example_label - current_dual;
    if (y_minus_a == 0.0) {
      // (y-a)*(log(y-a)-1) approaches 0 as y-a approaches 0.
      return 0.0;
    }
    if (y_minus_a < 0.0) {
      return std::numeric_limits<double>::max();
    }
    return y_minus_a * (log(y_minus_a) - 1) * example_weight;
  }

  double ComputePrimalLoss(const double wx, const double example_label,
                           const double example_weight) const final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSpoissonSLlossDTh mht_2(mht_2_v, 241, "", "./tensorflow/core/kernels/poisson-loss.h", "ComputePrimalLoss");

    return (exp(wx) - wx * example_label) * example_weight;
  }

  double PrimalLossDerivative(const double wx, const double label,
                              const double example_weight) const final {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSpoissonSLlossDTh mht_3(mht_3_v, 249, "", "./tensorflow/core/kernels/poisson-loss.h", "PrimalLossDerivative");

    return (exp(wx) - label) * example_weight;
  }

  // TODO(chapelle): We need to introduce a maximum_prediction parameter,
  // expose that parameter to the user and have this method return
  // 1.0/maximum_prediction.
  // Setting this at 1 for now, it only impacts the adaptive sampling.
  double SmoothnessConstant() const final {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSpoissonSLlossDTh mht_4(mht_4_v, 260, "", "./tensorflow/core/kernels/poisson-loss.h", "SmoothnessConstant");
 return 1; }

  Status ConvertLabel(float* const example_label) const final {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSpoissonSLlossDTh mht_5(mht_5_v, 265, "", "./tensorflow/core/kernels/poisson-loss.h", "ConvertLabel");

    if (*example_label < 0.0) {
      return errors::InvalidArgument(
          "Only non-negative labels can be used with the Poisson log loss. "
          "Found example with label: ", *example_label);
    }
    return Status::OK();
  }

 private:
  // One Newton step (see readme.md).
  double NewtonStep(const double x, const int num_loss_partitions,
                    const double label, const double wx,
                    const double example_weight,
                    const double weighted_example_norm,
                    const double current_dual) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSpoissonSLlossDTh mht_6(mht_6_v, 283, "", "./tensorflow/core/kernels/poisson-loss.h", "NewtonStep");

    const double expx = exp(x);
    const double numerator =
        x - wx - num_loss_partitions * weighted_example_norm *
        example_weight * (label - current_dual - expx);
    const double denominator =
       1 + num_loss_partitions * weighted_example_norm * example_weight * expx;
    return x - numerator / denominator;
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_LOGISTIC_LOSS_H_
