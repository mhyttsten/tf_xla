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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_COST_ANALYSIS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_COST_ANALYSIS_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTh() {
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


#include <functional>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

// HloCostAnalysis traverses an HLO graph and calculates the amount of
// computations required for the graph. Each HLO instruction handler provides
// the computation cost of the instruction, and the values are accumulated
// during the traversal for the entire graph. We treat normal floating point
// operations separately from transcendental operations.
class HloCostAnalysis : public ConstDfsHloVisitor {
 public:
  // Each HLO is associated to a vector of properties with the indices given
  // below. Sub-classes can add further properties.
  // MSVC 14.0 limitation requires the consts.
  typedef std::map<std::string, float, std::less<>> Properties;
  // shape_size is a function which returns the size in bytes of the top-level
  // buffer of a shape.
  using ShapeSizeFunction = std::function<int64_t(const Shape&)>;

  static constexpr const char kFlopsKey[] = "flops";
  static constexpr const char kTranscendentalsKey[] = "transcendentals";
  static constexpr const char kBytesAccessedKey[] = "bytes accessed";
  static constexpr const char kOptimalSecondsKey[] = "optimal_seconds";

  // A struct to encapsulate hardware-related options. This includes the shape
  // size function, which is used to encode hardware-specific padding and per
  // second rates of FLOPs, bytes per second (available bandwidth), and
  // transcendentals per second.
  struct Options {
    // Function which computes the size of the top-level of a given shape (not
    // including nested elements, if any). If null then bytes_accessed methods
    // return an error.
    ShapeSizeFunction shape_size;
    // How much of each property can be processed per second. E.g. if the
    // property is bytes accessed, this is the number of bytes that can be
    // processed per second. Is empty if no rates have been set.
    Properties per_second_rates = {};

    // Set the rates used to calculate the time taken by the computation.
    void set_flops_per_second(float value) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTh mht_0(mht_0_v, 238, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.h", "set_flops_per_second");

      per_second_rates[kFlopsKey] = value;
    }
    void set_transcendentals_per_second(float value) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTh mht_1(mht_1_v, 244, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.h", "set_transcendentals_per_second");

      per_second_rates[kTranscendentalsKey] = value;
    }
    void set_bytes_per_second(float value) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTh mht_2(mht_2_v, 250, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.h", "set_bytes_per_second");

      per_second_rates[kBytesAccessedKey] = value;
    }

    // Returns the specified per-second rate used by cost analysis.
    const float per_second_rate(const std::string& key) const {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTh mht_3(mht_3_v, 259, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.h", "per_second_rate");

      return GetProperty(key, per_second_rates);
    }
  };

  explicit HloCostAnalysis(const Options& options);
  explicit HloCostAnalysis(ShapeSizeFunction shape_size,
                           const Properties& per_second_rates = {});

  Status HandleElementwiseUnary(const HloInstruction* hlo) override;
  Status HandleElementwiseBinary(const HloInstruction* hlo) override;
  Status HandleConstant(const HloInstruction* constant) override;
  Status HandleIota(const HloInstruction* iota) override;
  Status HandleGetTupleElement(
      const HloInstruction* get_tuple_element) override;
  Status HandleSelect(const HloInstruction* hlo) override;
  Status HandleTupleSelect(const HloInstruction* hlo) override;
  Status HandleCompare(const HloInstruction* compare) override;
  Status HandleClamp(const HloInstruction* clamp) override;
  Status HandleReducePrecision(const HloInstruction* hlo) override;
  Status HandleConcatenate(const HloInstruction* concatenate) override;
  Status HandleAsyncStart(const HloInstruction* async_start) override;
  Status HandleAsyncUpdate(const HloInstruction* async_update) override;
  Status HandleAsyncDone(const HloInstruction* async_done) override;
  Status HandleCopyStart(const HloInstruction* send) override;
  Status HandleCopyDone(const HloInstruction* send_done) override;
  Status HandleSend(const HloInstruction* send) override;
  Status HandleSendDone(const HloInstruction* send_done) override;
  Status HandleRecv(const HloInstruction* recv) override;
  Status HandleRecvDone(const HloInstruction* recv_done) override;
  Status HandleConvert(const HloInstruction* convert) override;
  Status HandleCopy(const HloInstruction* copy) override;
  Status HandleDomain(const HloInstruction* domain) override;
  Status HandleDot(const HloInstruction* dot) override;
  Status HandleConvolution(const HloInstruction* convolution) override;
  Status HandleFft(const HloInstruction* fft) override;
  Status HandleTriangularSolve(const HloInstruction* hlo) override;
  Status HandleCholesky(const HloInstruction* hlo) override;
  Status HandleOptimizationBarrier(const HloInstruction* hlo) override;
  Status HandleAllGather(const HloInstruction* hlo) override;
  Status HandleAllGatherStart(const HloInstruction* hlo) override;
  Status HandleAllGatherDone(const HloInstruction* hlo) override;
  Status HandleAllReduce(const HloInstruction* crs) override;
  Status HandleReduceScatter(const HloInstruction* hlo) override;
  Status HandleAllReduceStart(const HloInstruction* hlo) override;
  Status HandleAllReduceDone(const HloInstruction* hlo) override;
  Status HandleAllToAll(const HloInstruction* hlo) override;
  Status HandleCollectivePermute(const HloInstruction* hlo) override;
  Status HandleCollectivePermuteStart(const HloInstruction* hlo) override;
  Status HandleCollectivePermuteDone(const HloInstruction* hlo) override;
  Status HandleReplicaId(const HloInstruction* hlo) override;
  Status HandlePartitionId(const HloInstruction* hlo) override;
  Status HandleInfeed(const HloInstruction* infeed) override;
  Status HandleOutfeed(const HloInstruction* outfeed) override;
  Status HandleRng(const HloInstruction* random) override;
  Status HandleRngBitGenerator(const HloInstruction* random) override;
  Status HandleRngGetAndUpdateState(const HloInstruction* random) override;
  Status HandleReverse(const HloInstruction* reverse) override;
  Status HandleSort(const HloInstruction* sort) override;
  Status HandleParameter(const HloInstruction* parameter) override;
  Status HandleReduce(const HloInstruction* reduce) override;
  Status HandleBatchNormTraining(
      const HloInstruction* batch_norm_training) override;
  Status HandleBatchNormInference(
      const HloInstruction* batch_norm_inference) override;
  Status HandleBatchNormGrad(const HloInstruction* batch_norm_grad) override;
  Status HandleFusion(const HloInstruction* fusion) override;
  Status HandleCall(const HloInstruction* call) override;
  Status HandleCustomCall(const HloInstruction* custom_call) override;
  Status HandleSlice(const HloInstruction* slice) override;
  Status HandleDynamicSlice(const HloInstruction* dynamic_slice) override;
  Status HandleDynamicUpdateSlice(
      const HloInstruction* dynamic_update_slice) override;
  Status HandleTuple(const HloInstruction* tuple) override;
  Status HandleMap(const HloInstruction* map) override;
  Status HandleReduceWindow(const HloInstruction* reduce_window) override;
  Status HandleSelectAndScatter(const HloInstruction* instruction) override;
  Status HandleBitcast(const HloInstruction* bitcast) override;
  Status HandleBroadcast(const HloInstruction* broadcast) override;
  Status HandlePad(const HloInstruction* pad) override;
  Status HandleReshape(const HloInstruction* reshape) override;
  Status HandleDynamicReshape(const HloInstruction* reshape) override;
  Status HandleAddDependency(const HloInstruction* add_dependency) override;
  Status HandleAfterAll(const HloInstruction* token) override;
  Status HandleTranspose(const HloInstruction* transpose) override;
  Status HandleWhile(const HloInstruction* xla_while) override;
  Status HandleConditional(const HloInstruction* conditional) override;
  Status HandleGather(const HloInstruction* gather) override;
  Status HandleScatter(const HloInstruction* scatter) override;
  Status HandleGetDimensionSize(const HloInstruction* get_size) override;
  Status HandleSetDimensionSize(const HloInstruction* set_size) override;
  Status FinishVisit(const HloInstruction* root) override;

  Status Preprocess(const HloInstruction* hlo) override;
  Status Postprocess(const HloInstruction* hlo) override;

  // Decorates shape_size_ by returning 0 immediately if the shape does not have
  // a layout.
  int64_t GetShapeSize(const Shape& shape) const;

  // Returns properties for the computation.
  float flop_count() const;
  float transcendental_count() const;
  float bytes_accessed() const;
  float optimal_seconds() const;

  // Returns the respective cost computed for a particular HLO instruction, or 0
  // if the HLO was not found to have a cost in the analysis.
  //
  // Note that the cost for sub HLO instructions are also returned if asked. For
  // example, body and condition of a while, fused instructions within a
  // fusion, or the add instruction of a reduce.
  int64_t flop_count(const HloInstruction& hlo) const;
  int64_t transcendental_count(const HloInstruction& hlo) const;
  int64_t bytes_accessed(const HloInstruction& hlo) const;
  int64_t operand_bytes_accessed(const HloInstruction& hlo, int64_t operand_num,
                                 ShapeIndex index = {}) const;
  int64_t output_bytes_accessed(const HloInstruction& hlo,
                                ShapeIndex index = {}) const;
  float optimal_seconds(const HloInstruction& hlo) const;

  // Get bytes read/written by this HLO. If memory_space is provided, it returns
  // the bytes read/written from/to the given memory space only.
  int64_t GetBytesRead(
      const HloInstruction& hlo,
      absl::optional<int64_t> memory_space = absl::nullopt) const;
  int64_t GetBytesWritten(
      const HloInstruction& hlo,
      absl::optional<int64_t> memory_space = absl::nullopt) const;

  const Properties& properties() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTh mht_4(mht_4_v, 392, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.h", "properties");
 return properties_sum_; }
  const float property(const std::string& key) const {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTh mht_5(mht_5_v, 397, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.h", "property");

    return GetProperty(key, properties());
  }

  // Returns the specified per-second rate used by cost analysis.
  const float per_second_rate(absl::string_view key) const {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("key: \"" + std::string(key.data(), key.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTh mht_6(mht_6_v, 406, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.h", "per_second_rate");

    return GetProperty(key, options_.per_second_rates);
  }

  // Return the key that is used to index into Properties for the specified
  // input/output at the shape index.
  static std::string GetOperandBytesAccessedKey(int64_t operand_num,
                                                ShapeIndex index = {});
  static std::string GetOutputBytesAccessedKey(ShapeIndex index = {});

  // Returns the estimated convolution flops.
  virtual int64_t GetConvolutionFlops(const HloInstruction* convolution);
  // Same as above but with parameters for shapes to allow for backends to
  // refine these.
  static int64_t GetConvolutionFlops(const HloInstruction* convolutions,
                                     const Shape& lhs_shape,
                                     const Shape& rhs_shape,
                                     const Shape& result_shape);

  // Returns the estimated dot flops.
  static int64_t GetDotFlops(const Shape& lhs_shape, const Shape& result_shape,
                             const DotDimensionNumbers& dnums);

 protected:
  typedef absl::flat_hash_map<const HloInstruction*, Properties>
      HloToProperties;

  // An FMA counts as two floating point operations in these analyzes.
  static constexpr int64_t kFmaFlops = 2;

  // Creates a nested instance of HloCostAnalysis using the same Options.
  virtual std::unique_ptr<HloCostAnalysis> CreateNestedCostAnalysis();

  // Returns the properties computed from visiting the computation rooted at the
  // given hlo. The cost of visited sub HLO instructions is saved to
  // hlo_properties_, which will be used by functions such as
  // flop_count(hlo_instruction) to return cost of a particular HLO instruction.
  StatusOr<Properties> ProcessSubcomputation(HloComputation* computation);

  // Utility function to handle all element-wise operations.
  Status HandleElementwiseOp(const HloInstruction* hlo_instruction);

  // Returns the default value if the key is not present in the
  // properties. Otherwise, returns the value that the key maps to from the
  // properties parameter.
  static float GetProperty(absl::string_view key, const Properties& properties,
                           float default_value = 0.0f);

  // Returns 0.0f if the hlo is not present in hlo_to_properties or if the key
  // is not present in hlo_to_properties[hlo]. Otherwise, returns the value that
  // the key maps to in the properties of the given hlo.
  static float GetPropertyForHlo(const HloInstruction& hlo,
                                 const std::string& key,
                                 const HloToProperties& hlo_to_properties);

  // Traverses a fusion operand to find the actual bytes accessed by the fusion
  // node.
  int64_t FusionParameterReadBytes(const HloInstruction* hlo) const;

  // Set bytes accessed by the specified operand and shape index.
  void SetOperandBytesAccessed(int64_t operand_num, float value);
  void SetOperandBytesAccessed(int64_t operand_num, ShapeIndex index,
                               float value);

  // Set bytes accessed by the output at the shape index.
  void SetOutputBytesAccessed(float value);
  void SetOutputBytesAccessed(ShapeIndex index, float value);

  HloToProperties hlo_properties_;

  // If true, the time taken will be computed from the rates for each property
  // and the total time will be the maximum time, which is the time of the
  // bottleneck.
  bool current_should_compute_bottleneck_time_;

  // The properties of the currently visited instruction. A HandleFoo method can
  // modify these to change the default values computed in Preprocess.
  Properties current_properties_;

  // The sum of the properties of all HLOs in the computation.
  Properties properties_sum_;

  // The hardware-specific options that contains things like the shape size
  // function and per-second rates.
  Options options_;

  HloCostAnalysis(const HloCostAnalysis&) = delete;
  HloCostAnalysis& operator=(const HloCostAnalysis&) = delete;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_COST_ANALYSIS_H_
