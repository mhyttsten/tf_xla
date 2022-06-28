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

#ifndef TENSORFLOW_CORE_KERNELS_HINGE_LOSS_H_
#define TENSORFLOW_CORE_KERNELS_HINGE_LOSS_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPShingeSLlossDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPShingeSLlossDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPShingeSLlossDTh() {
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


#include <algorithm>
#include <limits>

#include "tensorflow/core/kernels/loss.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

class HingeLossUpdater : public DualLossUpdater {
 public:
  // Computes the updated dual variable (corresponding) to a single example. The
  // updated dual value maximizes the objective function of the dual
  // optimization problem associated with hinge loss (conditioned on keeping the
  // rest of the dual variables intact). The method below finds an optimal delta
  // (difference between updated and previous dual value) using the update rule
  // within SDCA procedure (see http://arxiv.org/pdf/1209.1873v2.pdf, page 5)
  // and the particular form of conjugate function for hinge loss.
  //
  // The CoCoA+ modification is detailed in readme.md.
  //
  // TODO(sibyl-vie3Poto): Write up a doc with concrete derivation and point to it from
  // here.
  double ComputeUpdatedDual(const int num_loss_partitions, const double label,
                            const double example_weight,
                            const double current_dual, const double wx,
                            const double weighted_example_norm) const final {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPShingeSLlossDTh mht_0(mht_0_v, 214, "", "./tensorflow/core/kernels/hinge-loss.h", "ComputeUpdatedDual");

    // Intuitively there are 3 cases:
    // a. new optimal value of the dual variable falls within the admissible
    // range [0, 1]. In this case we set new dual to this value.
    // b. new optimal value is < 0. Then, because of convexity, the optimal
    // valid value for new dual = 0
    // c. new optimal value > 1.0. Then new optimal value should be set to 1.0.
    const double candidate_optimal_dual =
        current_dual + (label - wx) / (num_loss_partitions * example_weight *
                                       weighted_example_norm);
    if (label * candidate_optimal_dual < 0) {
      return 0.0;
    }
    if (label * candidate_optimal_dual > 1.0) {
      return label;
    }
    return candidate_optimal_dual;
  }

  // Conjugate of hinge loss. This is computed as:
  // \phi*(z) = z if z \in [-1, 0] and +infinity everywhere else. See for
  // instance http://www.eecs.berkeley.edu/~wainwrig/stat241b/lec10.pdf
  // Here we want the weighted version of the conjugate loss. It turns out, that
  // if w is the weight of an example, the conjugate of the weighted hinge loss
  // is given by:
  // \phi*(z) = z if z \in [-w, 0] and +infinity everywhere else. Here the
  // conjugate function depends not only on the weight of the example but also
  // on its label. In particular:
  // \phi_y*(z) = y*z if y*z \in [-w, 0] and +infinity everywhere else where
  // y \in {-1,1}. The following method implements \phi_y*(-\alpha/w).
  double ComputeDualLoss(const double current_dual, const double example_label,
                         const double example_weight) const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPShingeSLlossDTh mht_1(mht_1_v, 248, "", "./tensorflow/core/kernels/hinge-loss.h", "ComputeDualLoss");

    // For binary classification, there are 2 conjugate functions, one per
    // label value (-1 and 1).
    const double y_alpha = current_dual * example_label;  // y \alpha
    if (y_alpha < 0 || y_alpha > 1.0) {
      return std::numeric_limits<double>::max();
    }
    return -y_alpha * example_weight;
  }

  // Hinge loss for binary classification for a single example. Hinge loss
  // equals max(0, 1 - y * wx) (see https://en.wikipedia.org/wiki/Hinge_loss).
  // For weighted instances loss should be multiplied by the instance weight.
  double ComputePrimalLoss(const double wx, const double example_label,
                           const double example_weight) const final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPShingeSLlossDTh mht_2(mht_2_v, 265, "", "./tensorflow/core/kernels/hinge-loss.h", "ComputePrimalLoss");

    const double y_wx = example_label * wx;
    return std::max(0.0, 1 - y_wx) * example_weight;
  }

  double PrimalLossDerivative(const double wx, const double label,
                              const double example_weight) const final {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPShingeSLlossDTh mht_3(mht_3_v, 274, "", "./tensorflow/core/kernels/hinge-loss.h", "PrimalLossDerivative");

    if (label * wx < 1) {
      return -label * example_weight;
    }
    return 0;
  }

  // The smoothness constant is 0 since the derivative of the loss is not
  // Lipschitz
  double SmoothnessConstant() const final {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPShingeSLlossDTh mht_4(mht_4_v, 286, "", "./tensorflow/core/kernels/hinge-loss.h", "SmoothnessConstant");
 return 0; }

  // Converts binary example labels from 0.0 or 1.0 to -1.0 or 1.0 respectively
  // as expected by hinge loss.
  Status ConvertLabel(float* const example_label) const final {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPShingeSLlossDTh mht_5(mht_5_v, 293, "", "./tensorflow/core/kernels/hinge-loss.h", "ConvertLabel");

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
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_HINGE_LOSS_H_
