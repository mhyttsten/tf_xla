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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_ARITHMETIC_OPTIMIZER_TEST_UTILS_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_ARITHMETIC_OPTIMIZER_TEST_UTILS_H_
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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizer_test_utilsDTh {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizer_test_utilsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizer_test_utilsDTh() {
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


#include "tensorflow/core/grappler/optimizers/arithmetic_optimizer.h"
#include "tensorflow/core/grappler/optimizers/common_subgraph_elimination.h"
#include "tensorflow/core/grappler/optimizers/constant_folding.h"
#include "tensorflow/core/grappler/optimizers/model_pruner.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {
namespace grappler {

class ArithmeticOptimizerTest : public GrapplerTest {
 protected:
  // Optimize a graph using optimizer and prune all the nodes that no
  // longer have any output consumers.
  void OptimizeAndPrune(GraphOptimizer* optimizer, GrapplerItem* item,
                        GraphDef* output) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizer_test_utilsDTh mht_0(mht_0_v, 203, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer_test_utils.h", "OptimizeAndPrune");

    TF_EXPECT_OK(optimizer->Optimize(nullptr, *item, output));
    item->graph.Swap(output);
    output->Clear();
    TF_EXPECT_OK(ModelPruner().Optimize(nullptr, *item, output));
  }

  // Run optimizer twice to make sure the rewrite is idempotent.
  void DedupAndOptimizeTwiceAndPrune(GraphOptimizer* optimizer,
                                     GrapplerItem* item, GraphDef* output) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizer_test_utilsDTh mht_1(mht_1_v, 215, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer_test_utils.h", "DedupAndOptimizeTwiceAndPrune");

    TF_EXPECT_OK(CommonSubgraphElimination().Optimize(nullptr, *item, output));
    item->graph.Swap(output);
    output->Clear();
    TF_EXPECT_OK(optimizer->Optimize(nullptr, *item, output));
    item->graph.Swap(output);
    output->Clear();
    TF_EXPECT_OK(optimizer->Optimize(nullptr, *item, output));
    item->graph.Swap(output);
    output->Clear();
    TF_EXPECT_OK(ModelPruner().Optimize(nullptr, *item, output));
  }

  // Run optimizer twice to make sure the rewrite is idempotent.
  void OptimizeTwice(GraphOptimizer* optimizer, GrapplerItem* item,
                     GraphDef* output) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizer_test_utilsDTh mht_2(mht_2_v, 233, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer_test_utils.h", "OptimizeTwice");

    TF_EXPECT_OK(optimizer->Optimize(nullptr, *item, output));
    item->graph.Swap(output);
    output->Clear();
    TF_EXPECT_OK(optimizer->Optimize(nullptr, *item, output));
  }

  // Run optimizer twice to make sure the rewrite is idempotent.
  // Optionally run a constant folding pass before pruning.
  void OptimizeTwiceAndPrune(GraphOptimizer* optimizer, GrapplerItem* item,
                             GraphDef* output, bool const_folding = false) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizer_test_utilsDTh mht_3(mht_3_v, 246, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer_test_utils.h", "OptimizeTwiceAndPrune");

    TF_EXPECT_OK(optimizer->Optimize(nullptr, *item, output));

    item->graph.Swap(output);
    output->Clear();
    TF_EXPECT_OK(optimizer->Optimize(nullptr, *item, output));

    if (const_folding) {
      item->graph.Swap(output);
      output->Clear();
      TF_EXPECT_OK(ConstantFolding(/*cpu_device=*/nullptr)
                       .Optimize(nullptr, *item, output));
    }

    item->graph.Swap(output);
    output->Clear();
    TF_EXPECT_OK(ModelPruner().Optimize(nullptr, *item, output));
  }

  void DisableAddToAddNCombining(ArithmeticOptimizer* optimizer) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizer_test_utilsDTh mht_4(mht_4_v, 268, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer_test_utils.h", "DisableAddToAddNCombining");

    optimizer->options_.combine_add_to_addn = false;
  }

  void EnableOnlyAddToAddNCombining(ArithmeticOptimizer* optimizer) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizer_test_utilsDTh mht_5(mht_5_v, 275, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer_test_utils.h", "EnableOnlyAddToAddNCombining");

    DisableAllStages(optimizer);
    optimizer->options_.combine_add_to_addn = true;
  }

  void EnableOnlyFoldConjugateIntoTranspose(ArithmeticOptimizer* optimizer) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizer_test_utilsDTh mht_6(mht_6_v, 283, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer_test_utils.h", "EnableOnlyFoldConjugateIntoTranspose");

    DisableAllStages(optimizer);
    optimizer->options_.fold_conjugate_into_transpose = true;
  }

  void EnableOnlyFoldMultipleIntoConv(ArithmeticOptimizer* optimizer) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizer_test_utilsDTh mht_7(mht_7_v, 291, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer_test_utils.h", "EnableOnlyFoldMultipleIntoConv");

    DisableAllStages(optimizer);
    optimizer->options_.fold_multiply_into_conv = true;
  }

  void EnableOnlyFoldTransposeIntoMatMul(ArithmeticOptimizer* optimizer) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizer_test_utilsDTh mht_8(mht_8_v, 299, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer_test_utils.h", "EnableOnlyFoldTransposeIntoMatMul");

    DisableAllStages(optimizer);
    optimizer->options_.fold_transpose_into_matmul = true;
  }

  void EnableOnlyHoistCommonFactor(ArithmeticOptimizer* optimizer) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizer_test_utilsDTh mht_9(mht_9_v, 307, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer_test_utils.h", "EnableOnlyHoistCommonFactor");

    DisableAllStages(optimizer);
    optimizer->options_.hoist_common_factor_out_of_aggregation = true;
  }

  void EnableOnlyMinimizeBroadcasts(ArithmeticOptimizer* optimizer) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizer_test_utilsDTh mht_10(mht_10_v, 315, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer_test_utils.h", "EnableOnlyMinimizeBroadcasts");

    DisableAllStages(optimizer);
    optimizer->options_.minimize_broadcasts = true;
  }

  void EnableOnlyRemoveIdentityTranspose(ArithmeticOptimizer* optimizer) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizer_test_utilsDTh mht_11(mht_11_v, 323, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer_test_utils.h", "EnableOnlyRemoveIdentityTranspose");

    DisableAllStages(optimizer);
    optimizer->options_.remove_identity_transpose = true;
  }

  void EnableOnlyRemoveInvolution(ArithmeticOptimizer* optimizer) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizer_test_utilsDTh mht_12(mht_12_v, 331, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer_test_utils.h", "EnableOnlyRemoveInvolution");

    DisableAllStages(optimizer);
    optimizer->options_.remove_involution = true;
  }

  void EnableOnlyRemoveRedundantBitcast(ArithmeticOptimizer* optimizer) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizer_test_utilsDTh mht_13(mht_13_v, 339, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer_test_utils.h", "EnableOnlyRemoveRedundantBitcast");

    DisableAllStages(optimizer);
    optimizer->options_.remove_redundant_bitcast = true;
  }

  void EnableOnlyRemoveRedundantCast(ArithmeticOptimizer* optimizer) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizer_test_utilsDTh mht_14(mht_14_v, 347, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer_test_utils.h", "EnableOnlyRemoveRedundantCast");

    DisableAllStages(optimizer);
    optimizer->options_.remove_redundant_cast = true;
  }

  void EnableOnlyReduceUpsamplingDims(ArithmeticOptimizer* optimizer) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizer_test_utilsDTh mht_15(mht_15_v, 355, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer_test_utils.h", "EnableOnlyReduceUpsamplingDims");

    DisableAllStages(optimizer);
    optimizer->options_.reduce_upsampling_dims = true;
  }

  void EnableOnlyRemoveRedundantReshape(ArithmeticOptimizer* optimizer) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizer_test_utilsDTh mht_16(mht_16_v, 363, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer_test_utils.h", "EnableOnlyRemoveRedundantReshape");

    DisableAllStages(optimizer);
    optimizer->options_.remove_redundant_reshape = true;
  }

  void EnableOnlyRemoveNegation(ArithmeticOptimizer* optimizer) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizer_test_utilsDTh mht_17(mht_17_v, 371, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer_test_utils.h", "EnableOnlyRemoveNegation");

    DisableAllStages(optimizer);
    optimizer->options_.remove_negation = true;
  }

  void EnableOnlyReorderCastAndTranspose(ArithmeticOptimizer* optimizer) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizer_test_utilsDTh mht_18(mht_18_v, 379, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer_test_utils.h", "EnableOnlyReorderCastAndTranspose");

    DisableAllStages(optimizer);
    optimizer->options_.reorder_cast_like_and_value_preserving = true;
  }

  void EnableOnlyReplaceMulWithBroadcastByTile(ArithmeticOptimizer* optimizer) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizer_test_utilsDTh mht_19(mht_19_v, 387, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer_test_utils.h", "EnableOnlyReplaceMulWithBroadcastByTile");

    DisableAllStages(optimizer);
    optimizer->options_.replace_mul_with_tile = true;
  }

  void EnableOnlyReplaceMulWithSquare(ArithmeticOptimizer* optimizer) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizer_test_utilsDTh mht_20(mht_20_v, 395, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer_test_utils.h", "EnableOnlyReplaceMulWithSquare");

    DisableAllStages(optimizer);
    optimizer->options_.replace_mul_with_square = true;
  }

  void EnableOnlyReplacePackWithTileReshape(ArithmeticOptimizer* optimizer) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizer_test_utilsDTh mht_21(mht_21_v, 403, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer_test_utils.h", "EnableOnlyReplacePackWithTileReshape");

    DisableAllStages(optimizer);
    optimizer->options_.replace_pack_with_tile_reshape = true;
  }

  void EnableOnlyHoistCWiseUnaryChains(ArithmeticOptimizer* optimizer) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizer_test_utilsDTh mht_22(mht_22_v, 411, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer_test_utils.h", "EnableOnlyHoistCWiseUnaryChains");

    DisableAllStages(optimizer);
    optimizer->options_.hoist_cwise_unary_chains = true;
  }

  void EnableOnlySqrtDivToRsqrtMul(ArithmeticOptimizer* optimizer) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizer_test_utilsDTh mht_23(mht_23_v, 419, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer_test_utils.h", "EnableOnlySqrtDivToRsqrtMul");

    DisableAllStages(optimizer);
    optimizer->options_.convert_sqrt_div_to_rsqrt_mul = true;
  }

  void EnableOnlyLogSoftmax(ArithmeticOptimizer* optimizer) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizer_test_utilsDTh mht_24(mht_24_v, 427, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer_test_utils.h", "EnableOnlyLogSoftmax");

    DisableAllStages(optimizer);
    optimizer->options_.convert_log_softmax = true;
  }

  void EnableOnlyConvertPow(ArithmeticOptimizer* optimizer) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizer_test_utilsDTh mht_25(mht_25_v, 435, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer_test_utils.h", "EnableOnlyConvertPow");

    DisableAllStages(optimizer);
    optimizer->options_.convert_pow = true;
  }

  void EnableOnlyFuseSquaredDiff(ArithmeticOptimizer* optimizer) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizer_test_utilsDTh mht_26(mht_26_v, 443, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer_test_utils.h", "EnableOnlyFuseSquaredDiff");

    DisableAllStages(optimizer);
    optimizer->options_.fuse_squared_diff = true;
  }

  void EnableOnlyRemoveIdempotent(ArithmeticOptimizer* optimizer) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizer_test_utilsDTh mht_27(mht_27_v, 451, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer_test_utils.h", "EnableOnlyRemoveIdempotent");

    DisableAllStages(optimizer);
    optimizer->options_.remove_idempotent = true;
  }

  void EnableOnlyRemoveLogicalNot(ArithmeticOptimizer* optimizer) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizer_test_utilsDTh mht_28(mht_28_v, 459, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer_test_utils.h", "EnableOnlyRemoveLogicalNot");

    DisableAllStages(optimizer);
    optimizer->options_.remove_logical_not = true;
  }

  void EnableOnlySimplifyAggregation(ArithmeticOptimizer* optimizer) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizer_test_utilsDTh mht_29(mht_29_v, 467, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer_test_utils.h", "EnableOnlySimplifyAggregation");

    DisableAllStages(optimizer);
    optimizer->options_.simplify_aggregation = true;
  }

  void EnableOnlyLog1p(ArithmeticOptimizer* optimizer) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizer_test_utilsDTh mht_30(mht_30_v, 475, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer_test_utils.h", "EnableOnlyLog1p");

    DisableAllStages(optimizer);
    optimizer->options_.convert_log1p = true;
  }

  void EnableOnlyOptimizeMaxOrMinOfMonotonic(ArithmeticOptimizer* optimizer) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizer_test_utilsDTh mht_31(mht_31_v, 483, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer_test_utils.h", "EnableOnlyOptimizeMaxOrMinOfMonotonic");

    DisableAllStages(optimizer);
    optimizer->options_.optimize_max_or_min_of_monotonic = true;
  }

  void EnableOnlyExpm1(ArithmeticOptimizer* optimizer) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizer_test_utilsDTh mht_32(mht_32_v, 491, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer_test_utils.h", "EnableOnlyExpm1");

    DisableAllStages(optimizer);
    optimizer->options_.convert_expm1 = true;
  }

  void EnableOnlyUnaryOpsComposition(ArithmeticOptimizer* optimizer) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizer_test_utilsDTh mht_33(mht_33_v, 499, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer_test_utils.h", "EnableOnlyUnaryOpsComposition");

    DisableAllStages(optimizer);
    optimizer->options_.unary_ops_composition = true;
  }

  void EnableOnlyRemoveStackSliceSameAxis(ArithmeticOptimizer* optimizer) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizer_test_utilsDTh mht_34(mht_34_v, 507, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer_test_utils.h", "EnableOnlyRemoveStackSliceSameAxis");

    DisableAllStages(optimizer);
    optimizer->options_.remove_stack_slice_same_axis = true;
  }

  void EnableOnlySimplifyEmbeddingLookup(ArithmeticOptimizer* optimizer) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizer_test_utilsDTh mht_35(mht_35_v, 515, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer_test_utils.h", "EnableOnlySimplifyEmbeddingLookup");

    DisableAllStages(optimizer);
    optimizer->options_.simplify_embedding_lookup = true;
  }

  void EnableOnlyRemoveCastIntoSegmentReduction(
      ArithmeticOptimizer* optimizer) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizer_test_utilsDTh mht_36(mht_36_v, 524, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer_test_utils.h", "EnableOnlyRemoveCastIntoSegmentReduction");

    DisableAllStages(optimizer);
    optimizer->options_.remove_cast_into_segment_reduction = true;
  }

 private:
  void DisableAllStages(ArithmeticOptimizer* optimizer) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizer_test_utilsDTh mht_37(mht_37_v, 533, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer_test_utils.h", "DisableAllStages");

    ArithmeticOptimizer::ArithmeticOptimizerOptions options;
    options.dedup_computations = false;
    options.combine_add_to_addn = false;
    options.convert_sqrt_div_to_rsqrt_mul = false;
    options.convert_pow = false;
    options.convert_log1p = false;
    options.optimize_max_or_min_of_monotonic = false;
    options.fold_conjugate_into_transpose = false;
    options.fold_multiply_into_conv = false;
    options.fold_transpose_into_matmul = false;
    options.hoist_common_factor_out_of_aggregation = false;
    options.hoist_cwise_unary_chains = false;
    options.minimize_broadcasts = false;
    options.remove_identity_transpose = false;
    options.remove_involution = false;
    options.remove_idempotent = false;
    options.remove_redundant_bitcast = false;
    options.remove_redundant_cast = false;
    options.remove_redundant_reshape = false;
    options.remove_negation = false;
    options.remove_logical_not = false;
    options.reorder_cast_like_and_value_preserving = false;
    options.replace_mul_with_tile = false;
    options.replace_mul_with_square = false;
    options.simplify_aggregation = false;
    options.unary_ops_composition = false;
    options.simplify_embedding_lookup = false;
    options.remove_cast_into_segment_reduction = false;
    optimizer->options_ = options;
  }
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_ARITHMETIC_OPTIMIZER_TEST_UTILS_H_
