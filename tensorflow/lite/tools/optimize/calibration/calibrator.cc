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
class MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPScalibratorDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPScalibratorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPScalibratorDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/tools/optimize/calibration/calibrator.h"

#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/schema/schema_utils.h"
#include "tensorflow/lite/stderr_reporter.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/optimize/calibration/builtin_logging_ops/lstm.h"
#include "tensorflow/lite/tools/optimize/calibration/calibration_common.h"
#include "tensorflow/lite/tools/optimize/calibration/calibration_logger.h"
#include "tensorflow/lite/tools/optimize/calibration/calibration_reader.h"
#include "tensorflow/lite/tools/optimize/calibration/custom_logging_ops/lstm.h"
#include "tensorflow/lite/tools/optimize/calibration/logging_op.h"
#include "tensorflow/lite/tools/optimize/calibration/logging_op_resolver.h"

namespace tflite {
namespace optimize {
namespace calibration {

namespace {

// Calibrator is used to hold information that can be accessed during kernel
// invocations.
// TfLite kernel invocations are C functions and cannot look at the global
// structure of the graph. Calibrator allows the kernel invoke functions to
// access the global structure of graph and know which node is currently being
// executed. This also allows us to write a simple kernel invoke wrapper
// (see LoggingEval) that can work for most builtin ops.
class Calibrator {
 public:
  Calibrator(const std::unordered_map<const TfLiteNode*, OperatorInfo>&
                 node_ptr_opinfo_map,
             std::unique_ptr<LoggingOpResolver> logging_op_resolver,
             ErrorReporter* error_reporter)
      : node_ptr_opinfo_map_(node_ptr_opinfo_map),
        logging_op_resolver_(std::move(logging_op_resolver)),
        error_reporter_(error_reporter) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPScalibratorDTcc mht_0(mht_0_v, 238, "", "./tensorflow/lite/tools/optimize/calibration/calibrator.cc", "Calibrator");

    logger_ = absl::make_unique<Logger>();
  }

  // Returns the wrapped kernel invoke function |TfLiteRegistration.invoke|.
  KernelEvalFuncPtr GetKernelInvoke(const TfLiteNode* node) const;

  // Gets the instance of logger associated with the current context.
  Logger* GetLogger() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPScalibratorDTcc mht_1(mht_1_v, 249, "", "./tensorflow/lite/tools/optimize/calibration/calibrator.cc", "GetLogger");
 return logger_.get(); }

  // Gets the error reporter.
  ErrorReporter* GetErrorReporter() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPScalibratorDTcc mht_2(mht_2_v, 255, "", "./tensorflow/lite/tools/optimize/calibration/calibrator.cc", "GetErrorReporter");
 return error_reporter_; }

  // Gets the operator information about the given TfLiteNode.
  const OperatorInfo& GetOpInfo(const TfLiteNode* node) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPScalibratorDTcc mht_3(mht_3_v, 261, "", "./tensorflow/lite/tools/optimize/calibration/calibrator.cc", "GetOpInfo");

    return node_ptr_opinfo_map_.at(node);
  }

  std::vector<const TfLiteNode*> GetNodesUnderCalibration() {
    std::vector<const TfLiteNode*> nodes;
    for (const auto& entry : node_ptr_opinfo_map_) {
      nodes.push_back(entry.first);
    }
    return nodes;
  }

 private:
  std::unordered_map<const TfLiteNode*, OperatorInfo> node_ptr_opinfo_map_;
  std::unique_ptr<LoggingOpResolver> logging_op_resolver_;
  const std::unordered_map<int, OperatorInfo> index_opinfo_;
  std::unique_ptr<Logger> logger_;
  ErrorReporter* error_reporter_;
};

KernelEvalFuncPtr Calibrator::GetKernelInvoke(const TfLiteNode* node) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPScalibratorDTcc mht_4(mht_4_v, 284, "", "./tensorflow/lite/tools/optimize/calibration/calibrator.cc", "Calibrator::GetKernelInvoke");

  auto op_info = node_ptr_opinfo_map_.at(node);
  if (op_info.is_custom_op) {
    return logging_op_resolver_->GetWrappedKernelInvoke(op_info.name.c_str(),
                                                        op_info.version);
  }
  return logging_op_resolver_->GetWrappedKernelInvoke(op_info.builtin_op_code,
                                                      op_info.version);
}

// A registry of |Calibrator| objects per |TfLiteContext|.
// This global registry is needed to access |Calibrator| objects in the kernel
// invoke functions i.e. |TfLiteRegistration.invoke|.
// Kernel invoke functions are C functions that have limited access to
// |TfLiteContext|. Kernel invoke functions don't have access to global state of
// graph. That means during a kernel invocation, the function cannot know which
// node it was invoked for. E.g. in case of a model with |Conv| op at two
// locations, there is no easy way for the Conv.invoke function to disambiguate
// the calls.
//
// For calibration we solve this problem by creating a map of calibrators
// per |TfLiteContext|. This map is |GlobalCalibrationRegistry|.
//
// This registry is then accessed using a global getter function:
// |GetCalibratorRegistry|.
// E.g.
// TfLiteStatus SomeKernelInvokeFn(TfLiteContext* context, TfLiteNode* node) {
//   .... code ....
//   auto registry = GetCalibratorRegistry();
//   auto calibrator = registry->GetCalibrator(context);
//   ..... code ....
//  }
//
// This way the kernel invoke functions can get the access to the Calibrator
// object associated with the |TfLiteContext|.
class GlobalCalibratorRegistry {
 public:
  // Get the |Calibrator| associated with given context, returns null if no
  // calibrator is associated with the given context.
  Calibrator* GetCalibrator(const TfLiteNode* node) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPScalibratorDTcc mht_5(mht_5_v, 326, "", "./tensorflow/lite/tools/optimize/calibration/calibrator.cc", "GetCalibrator");

    if (node_to_calibrator_.find(node) == node_to_calibrator_.cend()) {
      return nullptr;
    }
    return node_to_calibrator_.at(node);
  }

  // Removes the association between calibrator and context.
  // Note: This deletes the calibrator as well.
  void RemoveCalibrator(const TfLiteContext* context) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPScalibratorDTcc mht_6(mht_6_v, 338, "", "./tensorflow/lite/tools/optimize/calibration/calibrator.cc", "RemoveCalibrator");

    Calibrator* calibrator = calibrator_registry_.at(context).get();
    auto nodes = calibrator->GetNodesUnderCalibration();
    for (auto node : nodes) {
      node_to_calibrator_.erase(node);
    }
    calibrator_registry_.erase(context);
  }

  // Creates an instance of |Calibrator|.
  // Registry owns the |Calibrator| object which can be deleted by calling
  // |RemoveCalibrator|.
  TfLiteStatus CreateCalibrator(
      const TfLiteContext* context,
      const std::unordered_map<const TfLiteNode*, OperatorInfo>& node_to_opinfo,
      std::unique_ptr<LoggingOpResolver> logging_op_resolver,
      Calibrator** calibrator_ptr, ErrorReporter* reporter) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPScalibratorDTcc mht_7(mht_7_v, 357, "", "./tensorflow/lite/tools/optimize/calibration/calibrator.cc", "CreateCalibrator");

    if (calibrator_registry_.find(context) != calibrator_registry_.cend()) {
      reporter->Report(
          "Failed to create calibrator, context already registered.");
      return kTfLiteError;
    }
    auto calibrator = absl::make_unique<Calibrator>(
        node_to_opinfo, std::move(logging_op_resolver), reporter);
    calibrator_registry_[context] = std::move(calibrator);
    *calibrator_ptr = calibrator_registry_.at(context).get();
    for (const auto& entry : node_to_opinfo) {
      node_to_calibrator_[entry.first] = *calibrator_ptr;
    }
    return kTfLiteOk;
  }

 private:
  absl::flat_hash_map<const TfLiteContext*, std::unique_ptr<Calibrator>>
      calibrator_registry_;
  absl::flat_hash_map<const TfLiteNode*, Calibrator*> node_to_calibrator_;
};

GlobalCalibratorRegistry* GetCalibratorRegistry() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPScalibratorDTcc mht_8(mht_8_v, 382, "", "./tensorflow/lite/tools/optimize/calibration/calibrator.cc", "GetCalibratorRegistry");

  static GlobalCalibratorRegistry* registry = new GlobalCalibratorRegistry();
  return registry;
}

// Get the logging kernel if there are any.
// TODO(jianlijianli): extend this to support multiple recipe for the same
// model.
logging_kernel_func_ptr GetLoggingEvalFunc(TfLiteContext* context,
                                           TfLiteNode* node,
                                           int builtin_op_code) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPScalibratorDTcc mht_9(mht_9_v, 395, "", "./tensorflow/lite/tools/optimize/calibration/calibrator.cc", "GetLoggingEvalFunc");

  switch (builtin_op_code) {
    case BuiltinOperator_LSTM: {
      if (node->intermediates->size == 12) {
        return tflite::optimize::calibration::custom::lstm_logging_kernel;
      }
      return tflite::optimize::calibration::builtin::lstm_logging_kernel;
    }
    case BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM:
      return tflite::optimize::calibration::builtin::
          unidirectional_sequence_lstm_logging_kernel;
    default:
      return nullptr;
  }
}

// A wrapper implementation for |TfLiteRegistration.invoke| that logs inputs,
// invokes the wrapped implementation and then logs the outputs.
TfLiteStatus LoggingEval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPScalibratorDTcc mht_10(mht_10_v, 416, "", "./tensorflow/lite/tools/optimize/calibration/calibrator.cc", "LoggingEval");

  Calibrator* calibrator = GetCalibratorRegistry()->GetCalibrator(node);

  if (!calibrator) {
    context->ReportError(context, "No calibrator found for context.");
    return kTfLiteError;
  }

  auto kernel_invoke = calibrator->GetKernelInvoke(node);
  auto logger = calibrator->GetLogger();
  auto op_info = calibrator->GetOpInfo(node);
  auto error_reporter = calibrator->GetErrorReporter();

  for (int i : op_info.loggable_inputs) {
    auto tensor = context->tensors[i];
    TF_LITE_ENSURE_STATUS(
        logger->LogTensorValue(op_info.subgraph_index, i, tensor.data.f,
                               tensor.bytes / sizeof(float), error_reporter));
  }
  auto builtin_op_code = calibrator->GetOpInfo(node).builtin_op_code;
  auto kernel_invoke_intermediate =
      GetLoggingEvalFunc(context, node, builtin_op_code);
  if (kernel_invoke_intermediate == nullptr) {
    TF_LITE_ENSURE_STATUS(kernel_invoke(context, node));
  } else {
    TF_LITE_ENSURE_STATUS(
        kernel_invoke_intermediate(context, op_info.subgraph_index, node,
                                   calibrator->GetLogger(), error_reporter));
  }

  // TODO(shashishekhar): An intermediate tensor in graph will get logged twice
  // once as an input and second time as output. This doesn't change the min max
  // values but is inefficient.
  // Using moving average will also break this.

  // Log input again to make sure the state tensors are captured after lstm
  // cell.
  for (int i : op_info.loggable_inputs) {
    auto tensor = context->tensors[i];
    TF_LITE_ENSURE_STATUS(
        logger->LogTensorValue(op_info.subgraph_index, i, tensor.data.f,
                               tensor.bytes / sizeof(float), error_reporter));
  }

  for (int i : op_info.loggable_outputs) {
    auto tensor = context->tensors[i];
    TF_LITE_ENSURE_STATUS(
        logger->LogTensorValue(op_info.subgraph_index, i, tensor.data.f,
                               tensor.bytes / sizeof(float), error_reporter));
  }

  return kTfLiteOk;
}

// Returns the loggable tensors. Not all inputs and outputs need to be logged.
// For example, const weight tensors which have buffers associated with them
// don't need to be logged.
std::vector<int> GetLoggableTensorIndices(
    const std::vector<int>& tensor_indices,
    const flatbuffers::Vector<flatbuffers::Offset<Tensor>>* tensors,
    const flatbuffers::Vector<flatbuffers::Offset<Buffer>>* tensor_buffers) {
  std::vector<int> loggable;
  for (auto tensor_index : tensor_indices) {
    if (tensor_index == kTfLiteOptionalTensor) {
      continue;
    }
    auto tensor = tensors->Get(tensor_index);
    auto buffer_index = tensor->buffer();
    const bool has_no_buffer =
        (tensor_buffers->Get(buffer_index) == nullptr) ||
        (tensor_buffers->Get(buffer_index)->data() == nullptr) ||
        (tensor_buffers->Get(buffer_index)->data()->size() == 0);
    if (has_no_buffer && tensor->type() == tflite::TensorType_FLOAT32) {
      loggable.push_back(tensor_index);
    }
  }
  return loggable;
}

// Creates a mapping between the static model graph and the runtime TfLiteNode*
// nodes in the graph for the given context.
// This is done by querying the TfLiteContext for node and registrations using
// the |NodeInfoDelegateObserver|.
TfLiteStatus GetNodeOpInfoMapAndContext(
    const absl::flat_hash_map<std::tuple<int, int>, OperatorInfo>&
        node_to_opinfo,
    tflite::Interpreter* const interpreter,
    std::unordered_map<const TfLiteNode*, OperatorInfo>* node_ptr_opinfo_map,
    TfLiteContext** context) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPScalibratorDTcc mht_11(mht_11_v, 507, "", "./tensorflow/lite/tools/optimize/calibration/calibrator.cc", "GetNodeOpInfoMapAndContext");

  *context = interpreter->primary_subgraph().context();

  // Since we only consider the primary subgraph while populating
  // node_to_opinfo, do the same here.
  // Because Flex delegate can merge multiple op nodes into one Delegate node if
  // they are located in a row, the size of the execution plan can be lesser
  // than the size of the graph's op nodes.
  TF_LITE_ENSURE(*context,
                 interpreter->execution_plan().size() <= node_to_opinfo.size());
  for (const auto& entry : node_to_opinfo) {
    auto op_info = entry.second;
    int subgraph_index, op_index;
    std::tie(subgraph_index, op_index) = entry.first;
    const auto* node_and_reg =
        interpreter->node_and_registration(subgraph_index, op_index);
    op_info.registration = &node_and_reg->second;
    node_ptr_opinfo_map->insert({&node_and_reg->first, op_info});
  }
  return kTfLiteOk;
}

string GetOpName(const tflite::OperatorCode& opcode) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPScalibratorDTcc mht_12(mht_12_v, 532, "", "./tensorflow/lite/tools/optimize/calibration/calibrator.cc", "GetOpName");

  if (opcode.custom_code() != nullptr) {
    return opcode.custom_code()->str();
  }
  return tflite::EnumNamesBuiltinOperator()[GetBuiltinCode(&opcode)];
}

// A |CalibrationReader| that owns the Calibrator.
class Reader : public CalibrationReader {
 public:
  Reader(const TfLiteContext* context, const Logger* logger)
      : CalibrationReader(logger), context_(context) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPScalibratorDTcc mht_13(mht_13_v, 546, "", "./tensorflow/lite/tools/optimize/calibration/calibrator.cc", "Reader");
}

  ~Reader() override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPScalibratorDTcc mht_14(mht_14_v, 551, "", "./tensorflow/lite/tools/optimize/calibration/calibrator.cc", "~Reader");
 GetCalibratorRegistry()->RemoveCalibrator(context_); }

 private:
  const TfLiteContext* context_;
};

bool HasInputs(BuiltinOperator code) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPScalibratorDTcc mht_15(mht_15_v, 560, "", "./tensorflow/lite/tools/optimize/calibration/calibrator.cc", "HasInputs");

  switch (code) {
    case BuiltinOperator_CALL_ONCE:
    case BuiltinOperator_VAR_HANDLE:
    // Custom ops, including Flex ops, might not have inputs.
    case BuiltinOperator_CUSTOM:
      return false;
    default:
      return true;
  }
}

bool HasOutputs(BuiltinOperator code) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPScalibratorDTcc mht_16(mht_16_v, 575, "", "./tensorflow/lite/tools/optimize/calibration/calibrator.cc", "HasOutputs");

  switch (code) {
    case BuiltinOperator_ASSIGN_VARIABLE:
    case BuiltinOperator_CALL_ONCE:
    // Custom ops, including Flex ops, might not have outputs.
    case BuiltinOperator_CUSTOM:
      return false;
    default:
      return true;
  }
}

}  // namespace

TfLiteStatus BuildLoggingInterpreter(
    const FlatBufferModel& model, const OpResolver& op_resolver,
    std::unique_ptr<Interpreter>* interpreter,
    std::unique_ptr<CalibrationReader>* calibration_reader) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPScalibratorDTcc mht_17(mht_17_v, 595, "", "./tensorflow/lite/tools/optimize/calibration/calibrator.cc", "BuildLoggingInterpreter");

  return BuildLoggingInterpreter(model.GetModel(), model.error_reporter(),
                                 op_resolver, interpreter, calibration_reader);
}

TfLiteStatus BuildLoggingInterpreter(
    const tflite::Model* tflite_model, ErrorReporter* error_reporter,
    const OpResolver& op_resolver, std::unique_ptr<Interpreter>* interpreter,
    std::unique_ptr<CalibrationReader>* calibration_reader) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPScalibratorDTcc mht_18(mht_18_v, 606, "", "./tensorflow/lite/tools/optimize/calibration/calibrator.cc", "BuildLoggingInterpreter");

  if (error_reporter == nullptr) {
    // Make sure error_reporter is valid.
    error_reporter = DefaultErrorReporter();
  }
  auto subgraphs = tflite_model->subgraphs();
  auto tensor_buffers = tflite_model->buffers();

  // Populate the node index to operator info map.
  // We want to collect this information so we can use it during runtime to
  // log details of which inputs and outputs.
  // At runtime TFLite kernel invoke functions can only look into their
  // own node in the graph (TFLiteNode*) and some limited context information.
  absl::flat_hash_map<std::tuple<int, int>, OperatorInfo> node_to_opinfo;
  BuiltinOpsSet builtin_op_and_versions;
  CustomOpsSet custom_op_and_versions;

  for (size_t subgraph_index = 0; subgraph_index < subgraphs->size();
       subgraph_index++) {
    auto subgraph = subgraphs->Get(subgraph_index);
    auto operator_codes = tflite_model->operator_codes();
    auto operators = subgraph->operators();
    auto tensors = subgraph->tensors();
    if (!operators) {
      continue;
    }

    for (size_t i = 0; i < operators->size(); i++) {
      OperatorInfo op_info;
      op_info.subgraph_index = subgraph_index;
      op_info.node_index = i;
      auto op = operators->Get(i);
      auto operator_code = operator_codes->Get(op->opcode_index());
      op_info.builtin_op_code = GetBuiltinCode(operator_code);
      op_info.name = GetOpName(*operator_code);
      op_info.is_custom_op = operator_code->custom_code() != nullptr;
      op_info.version = operator_code->version();

      auto op_inputs = op->inputs();
      auto op_outputs = op->outputs();
      if (op_inputs) {
        op_info.inputs = std::vector<int>(op_inputs->begin(), op_inputs->end());
      } else if (HasInputs(op_info.builtin_op_code)) {
        TFLITE_LOG(TFLITE_LOG_WARNING, "Op %s missing inputs",
                   op_info.name.c_str());
      }
      if (op_outputs) {
        op_info.outputs =
            std::vector<int>(op_outputs->begin(), op_outputs->end());
      } else if (HasOutputs(op_info.builtin_op_code)) {
        TFLITE_LOG(TFLITE_LOG_WARNING, "Op %s missing outputs",
                   op_info.name.c_str());
      }
      op_info.loggable_inputs =
          GetLoggableTensorIndices(op_info.inputs, tensors, tensor_buffers);
      op_info.loggable_outputs =
          GetLoggableTensorIndices(op_info.outputs, tensors, tensor_buffers);
      if (op_info.is_custom_op) {
        op_info.registration =
            op_resolver.FindOp(op_info.name.c_str(), operator_code->version());
        custom_op_and_versions.insert(
            {op_info.name.c_str(), operator_code->version()});
      } else {
        op_info.registration = op_resolver.FindOp(GetBuiltinCode(operator_code),
                                                  operator_code->version());
        builtin_op_and_versions.insert(
            {op_info.builtin_op_code, operator_code->version()});
      }
      std::tuple<int, int> key{subgraph_index, i};
      node_to_opinfo[key] = op_info;
    }
  }

  // Prepare the logging op resolver to use |LoggingEval| for kernel
  // invocations.
  auto logging_op_resolver = absl::make_unique<LoggingOpResolver>(
      builtin_op_and_versions, custom_op_and_versions, op_resolver, LoggingEval,
      error_reporter);
  tflite::InterpreterBuilder(tflite_model, *logging_op_resolver,
                             error_reporter)(interpreter);

  if (!(*interpreter)) {
    error_reporter->Report("Failed to construct interpreter");
    return kTfLiteError;
  }

  // Compute the mapping between runtime and static graph structure, i.e.
  // (TfLiteContext, TfLiteNode) -> OperatorInfo
  std::unordered_map<const TfLiteNode*, OperatorInfo> node_ptr_opinfo_map;
  TfLiteContext* context = nullptr;
  TF_LITE_ENSURE_STATUS(GetNodeOpInfoMapAndContext(
      node_to_opinfo, interpreter->get(), &node_ptr_opinfo_map, &context));

  Calibrator* calibrator = nullptr;
  // Register a calibrator object for the context. This can be accessed
  // during invocations by the logging kernels.
  TF_LITE_ENSURE_STATUS(GetCalibratorRegistry()->CreateCalibrator(
      context, node_ptr_opinfo_map, std::move(logging_op_resolver), &calibrator,
      error_reporter));
  *calibration_reader = std::unique_ptr<CalibrationReader>(
      new Reader(context, calibrator->GetLogger()));

  return kTfLiteOk;
}

}  // namespace calibration
}  // namespace optimize
}  // namespace tflite
