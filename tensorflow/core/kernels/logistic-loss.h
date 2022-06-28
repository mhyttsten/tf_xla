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

#ifndef TENSORFLOW_CORE_KERNELS_LOGISTIC_LOSS_H_
#define TENSORFLOW_CORE_KERNELS_LOGISTIC_LOSS_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSlogisticSLlossDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSlogisticSLlossDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSlogisticSLlossDTh() {
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

class LogisticLossUpdater : public DualLossUpdater {
 public:
  // Adding vs. Averaging in Distributed Primal-Dual Optimization.
  // Chenxin Ma, Virginia Smith, Martin Jaggi, Michael I. Jordan, Peter
  // Richtarik, Martin Takac http://arxiv.org/abs/1502.03508
  double ComputeUpdatedDual(const int num_loss_partitions, const double label,
                            const double example_weight,
                            const double current_dual, const double wx,
                            const double weighted_example_norm) const final {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlogisticSLlossDTh mht_0(mht_0_v, 203, "", "./tensorflow/core/kernels/logistic-loss.h", "ComputeUpdatedDual");

    // Newton algorithm converges quadratically so 10 steps will be largely
    // enough to achieve a very good precision
    static const int newton_total_steps = 10;
    double x = 0;
    for (int i = 0; i < newton_total_steps; ++i) {
      x = NewtonStep(x, num_loss_partitions, label, wx, example_weight,
                     weighted_example_norm, current_dual);
    }
    return 0.5 * (1 + tanh(x)) / label;
  }

  // Dual of logistic loss function.
  // https://en.wikipedia.org/wiki/Convex_conjugate
  double ComputeDualLoss(const double current_dual, const double example_label,
                         const double example_weight) const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlogisticSLlossDTh mht_1(mht_1_v, 221, "", "./tensorflow/core/kernels/logistic-loss.h", "ComputeDualLoss");

    // Dual of the logistic loss function is
    // ay * log(ay) + (1-ay) * log (1-ay), where a is the dual variable.
    const double ay = current_dual * example_label;
    const double log_ay = (ay > 0) ? log(ay) : 0;
    const double one_minus_ay = 1 - ay;
    const double log_one_minus_ay = (one_minus_ay > 0) ? log(one_minus_ay) : 0;
    return ((ay * log_ay) + (one_minus_ay * log_one_minus_ay)) * example_weight;
  }

  // Logistic loss for binary classification.
  // https://en.wikipedia.org/wiki/Loss_functions_for_classification
  double ComputePrimalLoss(const double wx, const double example_label,
                           const double example_weight) const final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlogisticSLlossDTh mht_2(mht_2_v, 237, "", "./tensorflow/core/kernels/logistic-loss.h", "ComputePrimalLoss");

    // Logistic loss:
    //   log(1 + e^(-ywx))
    //   log(e^0 + e^(-ywx))
    //   a + log(e^(0-a) + e^(-ywx - a)),  where a is max(0, -ywx)
    // https://hips.seas.harvard.edu/blog/2013/01/09/computing-log-sum-exp/
    const double y_wx = example_label * wx;
    if (y_wx > 0) {
      // 0 + log(e^(0) + e^(-ywx - 0))
      // log(1 + e^(-ywx))
      return log1p(exp(-y_wx)) * example_weight;
    }
    // -ywx + log(e^(ywx) + e^(-ywx + ywx))
    // log(e^(ywx) + e^(0)) - ywx
    // log(1 + e^(ywx)) - ywx
    return (log1p(exp(y_wx)) - y_wx) * example_weight;
  }

  // Derivative of logistic loss
  double PrimalLossDerivative(const double wx, const double label,
                              const double example_weight) const final {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlogisticSLlossDTh mht_3(mht_3_v, 260, "", "./tensorflow/core/kernels/logistic-loss.h", "PrimalLossDerivative");

    double inverse_exp_term = 0;
    if (label * wx > 0) {
      inverse_exp_term = exp(-label * wx) / (1 + exp(-label * wx));
    } else {
      inverse_exp_term = 1 / (1 + exp(label * wx));
    }
    return -inverse_exp_term * label * example_weight;
  }

  // The smoothness constant is 4 since the derivative of logistic loss, which
  // is exp(-x) / (1 + exp(-x)) can be shown to 0.25-Lipschitz (its derivative
  // is bounded by 0.25)
  double SmoothnessConstant() const final {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlogisticSLlossDTh mht_4(mht_4_v, 276, "", "./tensorflow/core/kernels/logistic-loss.h", "SmoothnessConstant");
 return 4; }

  // Converts binary example labels from 0.0 or 1.0 to -1.0 or 1.0 respectively
  // as expected by logistic regression.
  Status ConvertLabel(float* const example_label) const final {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlogisticSLlossDTh mht_5(mht_5_v, 283, "", "./tensorflow/core/kernels/logistic-loss.h", "ConvertLabel");

    if (*example_label == 0.0) {
      *example_label = -1;
      return Status::OK();
    }
    if (*example_label == 1.0) {
      return Status::OK();
    }
    return errors::InvalidArgument(
        "Only labels of 0.0 or 1.0 are supported right now. "
        "Found example with label: ",
        *example_label);
  }

 private:
  // We use Newton algorithm on a modified function (see readme.md).
  double NewtonStep(const double x, const int num_loss_partitions,
                    const double label, const double wx,
                    const double example_weight,
                    const double weighted_example_norm,
                    const double current_dual) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlogisticSLlossDTh mht_6(mht_6_v, 306, "", "./tensorflow/core/kernels/logistic-loss.h", "NewtonStep");

    const double tanhx = tanh(x);
    const double numerator = -2 * label * x - wx -
                             num_loss_partitions * weighted_example_norm *
                                 example_weight *
                                 (0.5 * (1 + tanhx) / label - current_dual);
    const double denominator =
        -2 * label - num_loss_partitions * weighted_example_norm *
                         example_weight * (1 - tanhx * tanhx) * 0.5 / label;
    return x - numerator / denominator;
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_LOGISTIC_LOSS_H_
