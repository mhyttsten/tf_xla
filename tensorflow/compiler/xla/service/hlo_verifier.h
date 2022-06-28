/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_VERIFIER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_VERIFIER_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_verifierDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_verifierDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_verifierDTh() {
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


#include <memory>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"

namespace xla {

// Visitor which verifies that the output shape is correctly set. Verifies
// against the inferred shape for the instruction.
class ShapeVerifier : public DfsHloVisitor {
 public:
  ShapeVerifier(bool layout_sensitive, bool allow_mixed_precision,
                std::function<int64_t(const Shape&)> shape_size_function)
      : layout_sensitive_(layout_sensitive),
        allow_mixed_precision_(allow_mixed_precision),
        shape_size_function_(shape_size_function) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_verifierDTh mht_0(mht_0_v, 204, "", "./tensorflow/compiler/xla/service/hlo_verifier.h", "ShapeVerifier");
}

  // Verifies that entry computation layout matches parameters and root shape of
  // the module's entry computation.
  virtual Status VerifyEntryComputationLayout(const HloModule& module);

  Status Preprocess(HloInstruction* hlo) override;

  Status HandleElementwiseUnary(HloInstruction* hlo) override;
  Status HandleElementwiseBinary(HloInstruction* hlo) override;
  Status HandleClamp(HloInstruction* clamp) override;
  Status HandleSelect(HloInstruction* select) override;
  Status HandleTupleSelect(HloInstruction* tuple_select) override;
  Status HandleConcatenate(HloInstruction* concatenate) override;
  Status HandleIota(HloInstruction* hlo) override;
  Status HandleConvert(HloInstruction* convert) override;
  Status HandleBitcastConvert(HloInstruction* convert) override;
  Status HandleCopy(HloInstruction* copy) override;
  Status HandleDot(HloInstruction* dot) override;
  Status HandleConvolution(HloInstruction* convolution) override;
  Status HandleFft(HloInstruction* fft) override;
  Status HandleCholesky(HloInstruction* hlo) override;
  Status HandleTriangularSolve(HloInstruction* hlo) override;
  Status HandleAllGather(HloInstruction* hlo) override;
  Status HandleAllGatherStart(HloInstruction* hlo) override;
  Status HandleAllGatherDone(HloInstruction* hlo) override;
  Status HandleAllReduce(HloInstruction* hlo) override;
  Status HandleAllReduceStart(HloInstruction* hlo) override;
  Status HandleAllReduceDone(HloInstruction* hlo) override;
  Status HandleAllToAll(HloInstruction* hlo) override;
  Status HandleCollectivePermute(HloInstruction* hlo) override;
  Status HandleCollectivePermuteStart(HloInstruction* hlo) override;
  Status HandleCollectivePermuteDone(HloInstruction* hlo) override;
  Status HandlePartitionId(HloInstruction* hlo) override;
  Status HandleReplicaId(HloInstruction* hlo) override;
  Status HandleReducePrecision(HloInstruction* reduce_precision) override;
  Status HandleInfeed(HloInstruction*) override;
  Status HandleOptimizationBarrier(HloInstruction* hlo) override;
  Status HandleOutfeed(HloInstruction*) override;
  Status HandleRng(HloInstruction*) override;
  Status HandleRngBitGenerator(HloInstruction*) override;
  Status HandleRngGetAndUpdateState(HloInstruction*) override;
  Status HandleReverse(HloInstruction* reverse) override;
  Status HandleSort(HloInstruction* sort) override;
  Status HandleConstant(HloInstruction* constant) override;
  Status HandleGetTupleElement(HloInstruction* get_tuple_element) override;
  Status HandleReduce(HloInstruction* reduce) override;
  Status HandleBitcast(HloInstruction* bitcast) override;
  Status HandleBroadcast(HloInstruction* broadcast) override;
  Status HandleReshape(HloInstruction* reshape) override;
  Status HandleDynamicReshape(HloInstruction* dynamic_reshape) override;
  Status HandleTranspose(HloInstruction* transpose) override;
  Status HandleParameter(HloInstruction*) override;
  Status HandleFusion(HloInstruction*) override;
  Status HandleCall(HloInstruction* call) override;
  Status HandleCustomCall(HloInstruction*) override;
  Status HandleSlice(HloInstruction* slice) override;
  Status HandleDynamicSlice(HloInstruction* dynamic_slice) override;
  Status HandleDynamicUpdateSlice(
      HloInstruction* dynamic_update_slice) override;
  Status HandleTuple(HloInstruction* tuple) override;
  Status HandleMap(HloInstruction* map) override;
  Status HandleReduceScatter(HloInstruction* hlo) override;
  Status HandleReduceWindow(HloInstruction* reduce_window) override;
  Status HandleSelectAndScatter(HloInstruction* instruction) override;
  Status HandleWhile(HloInstruction* xla_while) override;
  Status HandleConditional(HloInstruction* conditional) override;
  Status HandlePad(HloInstruction* pad) override;
  Status HandleAsyncStart(HloInstruction* async_start) override;
  Status HandleAsyncUpdate(HloInstruction* async_update) override;
  Status HandleAsyncDone(HloInstruction* async_done) override;
  Status HandleCopyStart(HloInstruction* copy_start) override;
  Status HandleCopyDone(HloInstruction* copy_done) override;
  Status HandleSend(HloInstruction* send) override;
  Status HandleSendDone(HloInstruction* send_done) override;
  Status HandleRecv(HloInstruction* recv) override;
  Status HandleRecvDone(HloInstruction* recv_done) override;
  Status HandleBatchNormTraining(HloInstruction* batch_norm_training) override;
  Status HandleBatchNormInference(
      HloInstruction* batch_norm_inference) override;
  Status HandleBatchNormGrad(HloInstruction* batch_norm_grad) override;
  Status HandleGather(HloInstruction* gather) override;
  Status HandleScatter(HloInstruction* scatter) override;
  Status HandleAfterAll(HloInstruction* token) override;
  Status HandleGetDimensionSize(HloInstruction* get_size) override;
  Status HandleSetDimensionSize(HloInstruction* set_size) override;
  Status HandleAddDependency(HloInstruction* add_dependency) override;

  Status FinishVisit(HloInstruction*) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_verifierDTh mht_1(mht_1_v, 295, "", "./tensorflow/compiler/xla/service/hlo_verifier.h", "FinishVisit");
 return Status::OK(); }

 protected:
  // Check the instruction's shape against the shape given by ShapeInference
  // and return an appropriate error if there is a mismatch.
  Status CheckShape(const HloInstruction* instruction,
                    const Shape& inferred_shape,
                    bool only_compare_minor_to_major_in_layout = false);

  // Overload which takes a StatusOr to reduce boilerplate in the caller.
  Status CheckShape(const HloInstruction* instruction,
                    const StatusOr<Shape>& inferred_shape_status);

  // Check a unary (binary, etc) instruction's shape against the inferred shape.
  Status CheckUnaryShape(const HloInstruction* instruction);
  Status CheckBinaryShape(const HloInstruction* instruction);
  Status CheckTernaryShape(const HloInstruction* instruction);
  Status CheckVariadicShape(const HloInstruction* instruction);

 private:
  // Helpers that switch on layout_sensitive_.
  bool ShapesSame(const Shape& a, const Shape& b,
                  bool minor_to_major_only = false,
                  bool ignore_memory_space = false) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_verifierDTh mht_2(mht_2_v, 321, "", "./tensorflow/compiler/xla/service/hlo_verifier.h", "ShapesSame");

    if (!layout_sensitive_) {
      return ShapeUtil::Compatible(a, b);
    }
    Shape::Equal equal;
    if (ignore_memory_space) {
      equal.IgnoreMemorySpaceInLayout();
    }
    if (minor_to_major_only) {
      equal.MinorToMajorOnlyInLayout();
    }
    return equal(a, b);
  }

  bool ShapesSameIgnoringFpPrecision(const Shape& a, const Shape& b,
                                     bool minor_to_major_only = false) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_verifierDTh mht_3(mht_3_v, 339, "", "./tensorflow/compiler/xla/service/hlo_verifier.h", "ShapesSameIgnoringFpPrecision");

    if (!layout_sensitive_) {
      return ShapeUtil::CompatibleIgnoringFpPrecision(a, b);
    }
    Shape::Equal equal;
    if (minor_to_major_only) {
      equal.MinorToMajorOnlyInLayout();
    }
    equal.IgnoreFpPrecision();
    return equal(a, b);
  }

  std::string StringifyShape(const Shape& s) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_verifierDTh mht_4(mht_4_v, 354, "", "./tensorflow/compiler/xla/service/hlo_verifier.h", "StringifyShape");

    return layout_sensitive_ ? ShapeUtil::HumanStringWithLayout(s)
                             : ShapeUtil::HumanString(s);
  }

  // Helpers that switch on allow_mixed_precision_.
  bool SameElementType(const Shape& a, const Shape& b) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_verifierDTh mht_5(mht_5_v, 363, "", "./tensorflow/compiler/xla/service/hlo_verifier.h", "SameElementType");

    return allow_mixed_precision_
               ? ShapeUtil::SameElementTypeIgnoringFpPrecision(a, b)
               : ShapeUtil::SameElementType(a, b);
  }

  // Checks that the given operand of the given instruction is of type TOKEN.
  Status CheckIsTokenOperand(const HloInstruction* instruction,
                             int64_t operand_no);

  // Checks that the shape of the given operand of the given instruction matches
  // the given parameter of the given computation.
  Status CheckOperandAndParameter(const HloInstruction* instruction,
                                  int64_t operand_number,
                                  const HloComputation* computation,
                                  int64_t parameter_number);

  // Returns true if the shapes of the two operands have the same element type,
  // and the result shape either has the same element type as the operand shapes
  // or mixed precision is allowed and the result shape and the operand shapes
  // have floating point element types.
  bool HasCompatibleElementTypes(const Shape& shape_0, const Shape& shape_1,
                                 const Shape& result_shape);

  // If the verifier is layout-sensitive, shapes must be equal to what's
  // expected.  Otherwise, the shapes must simply be compatible.
  bool layout_sensitive_;

  // Whether the inputs and output of an instruction can contain both F32s and
  // BF16s. Tuples that include both F32s and BF16s are allowed regardless of
  // this flag.
  bool allow_mixed_precision_;

  // Returns a target-specific shape size.
  std::function<int64_t(const Shape&)> shape_size_function_;
};

// An interface used to encapsulate target-specific verification quirks.
class TargetVerifierMetadata {
 public:
  explicit TargetVerifierMetadata(
      std::function<int64_t(const Shape&)> shape_size_function)
      : shape_size_function_(shape_size_function) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_verifierDTh mht_6(mht_6_v, 408, "", "./tensorflow/compiler/xla/service/hlo_verifier.h", "TargetVerifierMetadata");
}

  // Returns a target-specific shape size.
  int64_t ShapeSize(const Shape& shape) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_verifierDTh mht_7(mht_7_v, 414, "", "./tensorflow/compiler/xla/service/hlo_verifier.h", "ShapeSize");

    return shape_size_function_(shape);
  }

  void SetShapeSize(std::function<int64_t(const Shape&)> shape_size_function) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_verifierDTh mht_8(mht_8_v, 421, "", "./tensorflow/compiler/xla/service/hlo_verifier.h", "SetShapeSize");

    CHECK(shape_size_function_ == nullptr)
        << "shape_size_function_ is already set";
    shape_size_function_ = shape_size_function;
  }

  virtual std::unique_ptr<ShapeVerifier> GetVerifier() const = 0;

  virtual bool IsLayoutSensitive() const = 0;

  TargetVerifierMetadata() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_verifierDTh mht_9(mht_9_v, 434, "", "./tensorflow/compiler/xla/service/hlo_verifier.h", "TargetVerifierMetadata");
}
  virtual ~TargetVerifierMetadata() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_verifierDTh mht_10(mht_10_v, 438, "", "./tensorflow/compiler/xla/service/hlo_verifier.h", "~TargetVerifierMetadata");
}

  TargetVerifierMetadata(const TargetVerifierMetadata&) = delete;
  TargetVerifierMetadata& operator=(const TargetVerifierMetadata&) = delete;

 protected:
  // Returns a target-specific shape size.
  std::function<int64_t(const Shape&)> shape_size_function_;
};

// The default implementation of TargetVerifierMetadata, used unless the target
// needs to override it.
class DefaultVerifierMetadata : public TargetVerifierMetadata {
 public:
  DefaultVerifierMetadata(
      bool layout_sensitive, bool allow_mixed_precision,
      std::function<int64_t(const Shape&)> shape_size_function)
      : TargetVerifierMetadata(shape_size_function),
        layout_sensitive_(layout_sensitive),
        allow_mixed_precision_(allow_mixed_precision) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_verifierDTh mht_11(mht_11_v, 460, "", "./tensorflow/compiler/xla/service/hlo_verifier.h", "DefaultVerifierMetadata");
}

  // Creates a ShapeVerifier that checks that shapes match inferred
  // expectations. This creates a new verifier every time because ShapeVerifier,
  // being a DfsHloVisitor, is stateful. We want a clean object for each run of
  // the verifier.
  std::unique_ptr<ShapeVerifier> GetVerifier() const override {
    return absl::make_unique<ShapeVerifier>(
        layout_sensitive_, allow_mixed_precision_, shape_size_function_);
  }

  bool IsLayoutSensitive() const override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_verifierDTh mht_12(mht_12_v, 474, "", "./tensorflow/compiler/xla/service/hlo_verifier.h", "IsLayoutSensitive");
 return layout_sensitive_; }

 private:
  bool layout_sensitive_;
  bool allow_mixed_precision_;
};

// HLO pass that verifies invariants of HLO instructions for each computation in
// the module.
class HloVerifier : public HloModulePass {
 public:
  explicit HloVerifier(
      bool layout_sensitive, bool allow_mixed_precision,
      std::function<bool(const HloInstruction*)>
          instruction_can_change_layout_func = {},
      std::function<int64_t(const Shape&)> shape_size_func =
          [](const Shape& shape) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_verifierDTh mht_13(mht_13_v, 493, "", "./tensorflow/compiler/xla/service/hlo_verifier.h", "lambda");
 return ShapeUtil::ByteSizeOf(shape); })
      : target_metadata_(absl::make_unique<DefaultVerifierMetadata>(
            layout_sensitive, allow_mixed_precision, shape_size_func)),
        instruction_can_change_layout_func_(
            std::move(instruction_can_change_layout_func)),
        context_("Unknown") {
    CHECK(instruction_can_change_layout_func_ == nullptr || layout_sensitive);
  }

  // Uses custom target metadata
  explicit HloVerifier(std::unique_ptr<TargetVerifierMetadata> target_metadata,
                       absl::string_view context = "Unknown")
      : target_metadata_(std::move(target_metadata)), context_(context) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_verifierDTh mht_14(mht_14_v, 508, "", "./tensorflow/compiler/xla/service/hlo_verifier.h", "HloVerifier");
}

  ~HloVerifier() override = default;
  absl::string_view name() const override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_verifierDTh mht_15(mht_15_v, 514, "", "./tensorflow/compiler/xla/service/hlo_verifier.h", "name");
 return "verifier"; }

  // Never returns true; no instructions are ever modified by this pass.
  StatusOr<bool> Run(HloModule* module) override;

 private:
  std::unique_ptr<TargetVerifierMetadata> target_metadata_;

  // Determines whether an instruction can change layouts.
  std::function<bool(const HloInstruction*)>
      instruction_can_change_layout_func_;

  // The hlo pass when the verifier is invoked.
  std::string context_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_VERIFIER_H_
