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
class MHTracer_DTPStensorflowPScorePSkernelsPSloss_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSloss_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSloss_testDTcc() {
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

#include <limits>

#include "tensorflow/core/kernels/hinge-loss.h"
#include "tensorflow/core/kernels/logistic-loss.h"
#include "tensorflow/core/kernels/poisson-loss.h"
#include "tensorflow/core/kernels/smooth-hinge-loss.h"
#include "tensorflow/core/kernels/squared-loss.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

// TODO(sibyl-Aix6ihai): add a test to show the improvements of the Newton
// modification detailed in readme.md

// This test checks that the dual value after update is optimal.
// At the optimum the dual value should be the opposite of the primal gradient.
// This does not hold at a point where the primal is not differentiable.
void TestComputeUpdatedDual(const DualLossUpdater &loss_updater,
                            const int num_loss_partitions, const double label,
                            const double example_weight,
                            const double current_dual, const double wx,
                            const double weighted_example_norm) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSloss_testDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/kernels/loss_test.cc", "TestComputeUpdatedDual");

  double new_dual = loss_updater.ComputeUpdatedDual(
      num_loss_partitions, label, example_weight, current_dual, wx,
      weighted_example_norm);
  // The primal gradient needs to be computed after the weight update.
  double new_wx = wx + (new_dual - current_dual) * num_loss_partitions *
                           weighted_example_norm * example_weight;
  EXPECT_NEAR(new_dual, -loss_updater.PrimalLossDerivative(new_wx, label, 1.0),
              1e-5);
}

TEST(LogisticLoss, ComputePrimalLoss) {
  LogisticLossUpdater loss_updater;
  EXPECT_NEAR(0.693147,
              loss_updater.ComputePrimalLoss(0 /* wx */, 1 /* label */,
                                             1 /* example weight */),
              1e-3);
  EXPECT_NEAR(0.0,
              loss_updater.ComputePrimalLoss(70 /* wx */, 1 /* label */,
                                             1 /* example weight */),
              1e-3);
  EXPECT_NEAR(0.0,
              loss_updater.ComputePrimalLoss(-70 /* wx */, -1 /* label */,
                                             1 /* example weight */),
              1e-3);
}

TEST(LogisticLoss, ComputeDualLoss) {
  LogisticLossUpdater loss_updater;
  EXPECT_NEAR(0.0,
              loss_updater.ComputeDualLoss(0 /* current dual */, 1 /* label */,
                                           1 /* example weight */),
              1e-3);
  EXPECT_NEAR(0.0,
              loss_updater.ComputeDualLoss(1 /* current dual */, 1 /* label */,
                                           1 /* example weight */),
              1e-3);
  EXPECT_NEAR(
      -0.693147,
      loss_updater.ComputeDualLoss(0.5 /* current dual */, 1 /* label */,
                                   1 /* example weight */),
      1e-3);
}

TEST(LogisticLoss, ComputeUpdatedDual) {
  LogisticLossUpdater loss_updater;
  TestComputeUpdatedDual(loss_updater, 1 /* num partitions */, 1.0 /* label */,
                         1.0 /* example weight */, 0.5 /* current_dual */,
                         0.3 /* wx */, 10.0 /* weighted_example_norm */);
  TestComputeUpdatedDual(loss_updater, 2 /* num partitions */, -1.0 /* label */,
                         1.0 /* example weight */, 0.1 /* current_dual */,
                         -0.8 /* wx */, 10.0 /* weighted_example_norm */);
}

TEST(SquaredLoss, ComputePrimalLoss) {
  SquaredLossUpdater loss_updater;
  EXPECT_NEAR(0.5,
              loss_updater.ComputePrimalLoss(0.0 /* wx */, 1.0 /* label */,
                                             1.0 /* example weight */),
              1e-3);
  EXPECT_NEAR(40.5,
              loss_updater.ComputePrimalLoss(10.0 /* wx */, 1.0 /* label */,
                                             1.0 /* example weight */),
              1e-3);
  EXPECT_NEAR(0.125,
              loss_updater.ComputePrimalLoss(-0.5 /* wx */, -1.0 /* label */,
                                             1.0 /* example weight */),
              1e-3);
  EXPECT_NEAR(4.84,
              loss_updater.ComputePrimalLoss(1.2 /* wx */, -1.0 /* label */,
                                             2.0 /* example weight */),
              1e-3);
}

TEST(SquaredLoss, ComputeDualLoss) {
  SquaredLossUpdater loss_updater;
  EXPECT_NEAR(
      0.0,
      loss_updater.ComputeDualLoss(0.0 /* current dual */, -1.0 /* label */,
                                   1.0 /* example weight */),
      1e-3);
  EXPECT_NEAR(
      0.66,
      loss_updater.ComputeDualLoss(0.2 /* current dual */, -1.0 /* label */,
                                   3.0 /* example weight */),
      1e-3);
  EXPECT_NEAR(
      -0.375,
      loss_updater.ComputeDualLoss(1.5 /* current dual */, 1.0 /* label */,
                                   1.0 /* example weight */),
      1e-3);
  EXPECT_NEAR(
      -1.125,
      loss_updater.ComputeDualLoss(0.5 /* current dual */, 1.0 /* label */,
                                   3.0 /* example weight */),
      1e-3);
}

TEST(SquaredLoss, ComputeUpdatedDual) {
  SquaredLossUpdater loss_updater;
  TestComputeUpdatedDual(loss_updater, 1 /* num partitions */, 1.0 /* label */,
                         1.0 /* example weight */, 0.3 /* current_dual */,
                         0.3 /* wx */, 10.0 /* weighted_example_norm */);
  TestComputeUpdatedDual(loss_updater, 5 /* num partitions */, -1.0 /* label */,
                         1.0 /* example weight */, -0.4 /* current_dual */,
                         0.8 /* wx */, 10.0 /* weighted_example_norm */);
}

TEST(HingeLoss, ComputePrimalLoss) {
  HingeLossUpdater loss_updater;
  EXPECT_NEAR(1.0,
              loss_updater.ComputePrimalLoss(0.0 /* wx */, 1.0 /* label */,
                                             1.0 /* example weight */),
              1e-3);
  EXPECT_NEAR(0.0,
              loss_updater.ComputePrimalLoss(10.0 /* wx */, 1.0 /* label */,
                                             1.0 /* example weight */),
              1e-3);
  EXPECT_NEAR(0.5,
              loss_updater.ComputePrimalLoss(-0.5 /* wx */, -1.0 /* label */,
                                             1.0 /* example weight */),
              1e-3);
  EXPECT_NEAR(4.4,
              loss_updater.ComputePrimalLoss(1.2 /* wx */, -1.0 /* label */,
                                             2.0 /* example weight */),
              1e-3);
}

TEST(HingeLoss, ComputeDualLoss) {
  HingeLossUpdater loss_updater;
  EXPECT_NEAR(
      0.0,
      loss_updater.ComputeDualLoss(0.0 /* current dual */, -1.0 /* label */,
                                   1.0 /* example weight */),
      1e-3);
  EXPECT_NEAR(
      std::numeric_limits<double>::max(),
      loss_updater.ComputeDualLoss(0.2 /* current dual */, -1.0 /* label */,
                                   3.0 /* example weight */),
      1e-3);
  EXPECT_NEAR(
      std::numeric_limits<double>::max(),
      loss_updater.ComputeDualLoss(1.5 /* current dual */, 1.0 /* label */,
                                   1.0 /* example weight */),
      1e-3);
  EXPECT_NEAR(
      -1.5,
      loss_updater.ComputeDualLoss(0.5 /* current dual */, 1.0 /* label */,
                                   3.0 /* example weight */),
      1e-3);
}

TEST(HingeLoss, ConvertLabel) {
  HingeLossUpdater loss_updater;
  float example_label = 1.0;
  Status status;

  // A label with value 1.0 should remain intact.
  TF_EXPECT_OK(loss_updater.ConvertLabel(&example_label));
  EXPECT_EQ(1.0, example_label);

  // A label with value 0.0 should be converted to -1.0.
  example_label = 0.0;
  TF_EXPECT_OK(loss_updater.ConvertLabel(&example_label));
  EXPECT_EQ(-1.0, example_label);

  // Any other initial value should throw an error.
  example_label = 0.5;
  status = loss_updater.ConvertLabel(&example_label);
  EXPECT_FALSE(status.ok());
}

TEST(HingeLoss, ComputeUpdatedDual) {
  HingeLossUpdater loss_updater;
  // For the two tests belows, y*wx=1 after the update which is a
  // non-differentiable point of the hinge loss and TestComputeUpdatedDual
  // cannot be used. Check value of the dual variable instead.
  EXPECT_NEAR(0.507,
              loss_updater.ComputeUpdatedDual(
                  1 /* num partitions */, 1.0 /* label */,
                  1.0 /* example weight */, 0.5 /* current_dual */,
                  0.3 /* wx */, 100.0 /* weighted_example_norm */),
              1e-3);
  EXPECT_NEAR(-0.416,
              loss_updater.ComputeUpdatedDual(
                  10 /* num partitions */, -1.0 /* label */,
                  1.0 /* example weight */, -0.4 /* current_dual */,
                  0.6 /* wx */, 10.0 /* weighted_example_norm */),
              1e-3);
  TestComputeUpdatedDual(loss_updater, 1 /* num partitions */, 1.0 /* label */,
                         1.0 /* example weight */, -0.5 /* current_dual */,
                         0.3 /* wx */, 10.0 /* weighted_example_norm */);
  TestComputeUpdatedDual(loss_updater, 1 /* num partitions */, -1.0 /* label */,
                         2.0 /* example weight */, -1.0 /* current_dual */,
                         0.3 /* wx */, 10.0 /* weighted_example_norm */);
}

TEST(SmoothHingeLoss, ComputePrimalLoss) {
  SmoothHingeLossUpdater loss_updater;
  EXPECT_NEAR(0.5,
              loss_updater.ComputePrimalLoss(0.0 /* wx */, 1.0 /* label */,
                                             1.0 /* example weight */),
              1e-3);
  EXPECT_NEAR(0.0,
              loss_updater.ComputePrimalLoss(10.0 /* wx */, 1.0 /* label */,
                                             1.0 /* example weight */),
              1e-3);
  EXPECT_NEAR(0.125,
              loss_updater.ComputePrimalLoss(-0.5 /* wx */, -1.0 /* label */,
                                             1.0 /* example weight */),
              1e-3);
  EXPECT_NEAR(3.4,
              loss_updater.ComputePrimalLoss(1.2 /* wx */, -1.0 /* label */,
                                             2.0 /* example weight */),
              1e-3);
}

TEST(SmoothHingeLoss, ComputeDualLoss) {
  SmoothHingeLossUpdater loss_updater;
  EXPECT_NEAR(
      0.0,
      loss_updater.ComputeDualLoss(0.0 /* current dual */, -1.0 /* label */,
                                   1.0 /* example weight */),
      1e-3);
  EXPECT_NEAR(
      std::numeric_limits<double>::max(),
      loss_updater.ComputeDualLoss(0.2 /* current dual */, -1.0 /* label */,
                                   3.0 /* example weight */),
      1e-3);
  EXPECT_NEAR(
      std::numeric_limits<double>::max(),
      loss_updater.ComputeDualLoss(1.5 /* current dual */, 1.0 /* label */,
                                   1.0 /* example weight */),
      1e-3);
  EXPECT_NEAR(
      -1.125,
      loss_updater.ComputeDualLoss(0.5 /* current dual */, 1.0 /* label */,
                                   3.0 /* example weight */),
      1e-3);
}

TEST(SmoothHingeLoss, ComputeUpdatedDual) {
  SmoothHingeLossUpdater loss_updater;
  TestComputeUpdatedDual(loss_updater, 1 /* num partitions */, 1.0 /* label */,
                         1.0 /* example weight */, 0.3 /* current_dual */,
                         0.3 /* wx */, 10.0 /* weighted_example_norm */);
  TestComputeUpdatedDual(loss_updater, 5 /* num partitions */, -1.0 /* label */,
                         1.0 /* example weight */, -0.4 /* current_dual */,
                         0.8 /* wx */, 10.0 /* weighted_example_norm */);
}

TEST(PoissonLoss, ComputePrimalLoss) {
  PoissonLossUpdater loss_updater;
  EXPECT_NEAR(1.0,
              loss_updater.ComputePrimalLoss(0.0 /* wx */, 3.0 /* label */,
                                             1.0 /* example weight */),
              1e-3);
  EXPECT_NEAR(21996.0,
              loss_updater.ComputePrimalLoss(10.0 /* wx */, 3.0 /* label */,
                                             1.0 /* example weight */),
              1.0);
  EXPECT_NEAR(0.606,
              loss_updater.ComputePrimalLoss(-0.5 /* wx */, 0.0 /* label */,
                                             1.0 /* example weight */),
              1e-3);
  EXPECT_NEAR(6.64,
              loss_updater.ComputePrimalLoss(1.2 /* wx */, 0.0 /* label */,
                                             2.0 /* example weight */),
              1e-2);
}

TEST(PoissonLoss, ComputeDualLoss) {
  PoissonLossUpdater loss_updater;
  // Dual is undefined.
  EXPECT_NEAR(
      std::numeric_limits<double>::max(),
      loss_updater.ComputeDualLoss(1.0 /* current dual */, 0.0 /* label */,
                                   1.0 /* example weight */),
      1e-3);
  EXPECT_NEAR(
      0.0,
      loss_updater.ComputeDualLoss(0.0 /* current dual */, 0.0 /* label */,
                                   3.0 /* example weight */),
      1e-3);
  EXPECT_NEAR(
      -0.847,
      loss_updater.ComputeDualLoss(1.5 /* current dual */, 2.0 /* label */,
                                   1.0 /* example weight */),
      1e-3);
  EXPECT_NEAR(
      -2.675,
      loss_updater.ComputeDualLoss(0.5 /* current dual */, 2.0 /* label */,
                                   3.0 /* example weight */),
      1e-3);
}

TEST(PoissonLoss, ConvertLabel) {
  PoissonLossUpdater loss_updater;
  float example_label = -1.0;
  // Negative label should throw an error.
  Status status = loss_updater.ConvertLabel(&example_label);
  EXPECT_FALSE(status.ok());
}

TEST(PoissonLoss, ComputeUpdatedDual) {
  PoissonLossUpdater loss_updater;
  TestComputeUpdatedDual(loss_updater, 1 /* num partitions */, 2.0 /* label */,
                         1.0 /* example weight */, 0.5 /* current_dual */,
                         0.3 /* wx */, 10.0 /* weighted_example_norm */);
  TestComputeUpdatedDual(loss_updater, 2 /* num partitions */, 0.0 /* label */,
                         1.0 /* example weight */, 0.0 /* current_dual */,
                         -0.8 /* wx */, 10.0 /* weighted_example_norm */);
}

}  // namespace
}  // namespace tensorflow
