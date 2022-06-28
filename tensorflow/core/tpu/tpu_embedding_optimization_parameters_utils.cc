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
class MHTracer_DTPStensorflowPScorePStpuPStpu_embedding_optimization_parameters_utilsDTcc {
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
   MHTracer_DTPStensorflowPScorePStpuPStpu_embedding_optimization_parameters_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPStpu_embedding_optimization_parameters_utilsDTcc() {
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

#include "tensorflow/core/tpu/tpu_embedding_optimization_parameters_utils.h"

#include <string>
#include <utility>

#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/stringprintf.h"

namespace tensorflow {
namespace tpu {

std::string GetOptimizationAlgorithmName(OptimizationAlgorithm alg) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStpuPStpu_embedding_optimization_parameters_utilsDTcc mht_0(mht_0_v, 201, "", "./tensorflow/core/tpu/tpu_embedding_optimization_parameters_utils.cc", "GetOptimizationAlgorithmName");

  switch (alg) {
    case OptimizationAlgorithm::kAdagrad:
      return "Adagrad";
    case OptimizationAlgorithm::kAdagradMomentum:
      return "AdagradMomentum";
    case OptimizationAlgorithm::kBoundedAdagrad:
      return "BoundedAdagrad";
    case OptimizationAlgorithm::kStochasticGradientDescent:
      return "StochasticGradientDescent";
    case OptimizationAlgorithm::kFtrl:
      return "FTRL";
    case OptimizationAlgorithm::kAdam:
      return "ADAM";
    case OptimizationAlgorithm::kMomentum:
      return "Momentum";
    case OptimizationAlgorithm::kRmsProp:
      return "RMSProp";
    case OptimizationAlgorithm::kCenteredRmsProp:
      return "CenteredRMSProp";
    case OptimizationAlgorithm::kMdlAdagradLight:
      return "MDLAdagradLight";
    case OptimizationAlgorithm::kAdadelta:
      return "Adadelta";
    case OptimizationAlgorithm::kProximalAdagrad:
      return "ProximalAdagrad";
    case OptimizationAlgorithm::kOnlineYogi:
      return "OnlineYogi";
    case OptimizationAlgorithm::kProximalYogi:
      return "ProximalYogi";
    case OptimizationAlgorithm::kFrequencyEstimator:
      return "FrequencyEstimator";
    case OptimizationAlgorithm::kUserDefinedProgram:
      return "UserDefinedProgram";
    case OptimizationAlgorithm::kAssign:
      return "Assign";
    case OptimizationAlgorithm::PARAMETERS_NOT_SET:
      return "*** Not set ***";
  }
  return "*** Not set ***";
}

std::string GetOptimizationAlgorithmFriendlyName(OptimizationAlgorithm alg) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStpuPStpu_embedding_optimization_parameters_utilsDTcc mht_1(mht_1_v, 246, "", "./tensorflow/core/tpu/tpu_embedding_optimization_parameters_utils.cc", "GetOptimizationAlgorithmFriendlyName");

  switch (alg) {
    case OptimizationAlgorithm::kAdagrad:
      return "Adagrad";
    case OptimizationAlgorithm::kAdagradMomentum:
      return "Adagrad with Momentum";
    case OptimizationAlgorithm::kBoundedAdagrad:
      return "Bounded Adagrad";
    case OptimizationAlgorithm::kStochasticGradientDescent:
      return "stochastic gradient descent";
    case OptimizationAlgorithm::kFtrl:
      return "FTRL";
    case OptimizationAlgorithm::kAdam:
      return "ADAM";
    case OptimizationAlgorithm::kMomentum:
      return "Momentum";
    case OptimizationAlgorithm::kRmsProp:
      return "RMSProp";
    case OptimizationAlgorithm::kCenteredRmsProp:
      return "centered RMSProp";
    case OptimizationAlgorithm::kMdlAdagradLight:
      return "MDL Adagrad Light";
    case OptimizationAlgorithm::kAdadelta:
      return "Adadelta";
    case OptimizationAlgorithm::kProximalAdagrad:
      return "proximal Adagrad";
    case OptimizationAlgorithm::kOnlineYogi:
      return "online Yogi";
    case OptimizationAlgorithm::kProximalYogi:
      return "proximal Yogi";
    case OptimizationAlgorithm::kFrequencyEstimator:
      return "frequency estimator";
    case OptimizationAlgorithm::kUserDefinedProgram:
      return "UserDefinedProgram";
    case OptimizationAlgorithm::kAssign:
      return "Assign";
    case OptimizationAlgorithm::PARAMETERS_NOT_SET:
      return "unknown (not specified)";
  }
  return "unknown (not specified)";
}

// Returns the number of optimization parameter vectors used by the optimization
// algorithm, excluding the weights themselves and assuming no gradient
// accumulation.
Status GetBaseAuxiliaryParameterCount(const OptimizationParameters& params,
                                      int* count) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStpuPStpu_embedding_optimization_parameters_utilsDTcc mht_2(mht_2_v, 295, "", "./tensorflow/core/tpu/tpu_embedding_optimization_parameters_utils.cc", "GetBaseAuxiliaryParameterCount");

  switch (params.parameters_case()) {
    case OptimizationAlgorithm::kAdagrad:
      *count = 1;
      return Status::OK();
    case OptimizationAlgorithm::kAdagradMomentum:
      *count = 2;
      return Status::OK();
    case OptimizationAlgorithm::kBoundedAdagrad:
      *count = 1;
      return Status::OK();
    case OptimizationAlgorithm::kStochasticGradientDescent:
      *count = 0;
      return Status::OK();
    case OptimizationAlgorithm::kFtrl:
      *count = 2;
      return Status::OK();
    case OptimizationAlgorithm::kAdam:
      *count = 2;
      return Status::OK();
    case OptimizationAlgorithm::kMomentum:
      *count = 1;
      return Status::OK();
    case OptimizationAlgorithm::kRmsProp:
      *count = 2;
      return Status::OK();
    case OptimizationAlgorithm::kCenteredRmsProp:
      *count = 3;
      return Status::OK();
    case OptimizationAlgorithm::kMdlAdagradLight:
      *count = 3;
      return Status::OK();
    case OptimizationAlgorithm::kAdadelta:
      *count = 2;
      return Status::OK();
    case OptimizationAlgorithm::kProximalAdagrad:
      *count = 1;
      return Status::OK();
    case OptimizationAlgorithm::kOnlineYogi:
      *count = 2;
      return Status::OK();
    case OptimizationAlgorithm::kProximalYogi:
      *count = 2;
      return Status::OK();
    case OptimizationAlgorithm::kFrequencyEstimator:
      *count = 1;
      return Status::OK();
    case OptimizationAlgorithm::kUserDefinedProgram: {
      const xla::ProgramShapeProto& program_shape =
          params.user_defined_program().program().host_program_shape();

      const int num_inputs = program_shape.parameters_size();
      const int num_outputs = program_shape.result().tuple_shapes_size();

      if ((num_inputs < 2) || ((num_inputs != num_outputs + 1) &&
                               (num_inputs != num_outputs + 2))) {
        return errors::InvalidArgument(
            "User-defined TPU embedding optimizer program must have at least "
            "two inputs and the number of outputs must be 1 or 2 less than the "
            "number of inputs. Received ",
            num_inputs, " input(s) and ", num_outputs, "output(s).");
      }

      *count = num_outputs - 1;

      return Status::OK();
    }
    case OptimizationAlgorithm::kAssign:
      *count = 0;
      return Status::OK();
    case OptimizationAlgorithm::PARAMETERS_NOT_SET:
      return errors::InvalidArgument("No optimization algorithm specified");
  }
  return errors::InvalidArgument("No optimization algorithm specified");
}

Status GetGradientAccumulationSupport(const OptimizationParameters& params,
                                      GradientAccumulationSupport* support) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStpuPStpu_embedding_optimization_parameters_utilsDTcc mht_3(mht_3_v, 375, "", "./tensorflow/core/tpu/tpu_embedding_optimization_parameters_utils.cc", "GetGradientAccumulationSupport");

  int auxiliary_parameter_count;
  TF_RETURN_IF_ERROR(
      GetBaseAuxiliaryParameterCount(params, &auxiliary_parameter_count));
  *support = auxiliary_parameter_count + 1 <= kMaxAuxiliaryParameterCount
                 ? GradientAccumulationSupport::kSupported
                 : GradientAccumulationSupport::kNotSupported;
  return Status::OK();
}

Status UseGradientAccumulation(const OptimizationParameters& params,
                               bool* use_gradient_accumulation) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePStpuPStpu_embedding_optimization_parameters_utilsDTcc mht_4(mht_4_v, 389, "", "./tensorflow/core/tpu/tpu_embedding_optimization_parameters_utils.cc", "UseGradientAccumulation");

  GradientAccumulationSupport support;
  TF_RETURN_IF_ERROR(GetGradientAccumulationSupport(params, &support));
  bool raw_gradient_accumulation_status = false;
  switch (params.gradient_accumulation_status()) {
    case GradientAccumulationStatus::UNSPECIFIED: {
      // Default is now to turn gradient accumulation on by default.
      raw_gradient_accumulation_status = true;
      break;
    }
    case GradientAccumulationStatus::DISABLED: {
      raw_gradient_accumulation_status = false;
      break;
    }
    case GradientAccumulationStatus::ENABLED: {
      raw_gradient_accumulation_status = true;
      break;
    }
    default:
      return errors::Internal(
          absl::StrCat("Unsupported gradient accumulation status ",
                       GradientAccumulationStatus_Status_Name(
                           params.gradient_accumulation_status())));
  }
  switch (support) {
    case GradientAccumulationSupport::kSupported: {
      *use_gradient_accumulation = raw_gradient_accumulation_status;
      break;
    }
    case GradientAccumulationSupport::kNotSupported: {
      if (raw_gradient_accumulation_status) {
        return errors::InvalidArgument(strings::Printf(
            "Optimization algorithm %s does not support gradient accumulation "
            "but parameters specify it.",
            GetOptimizationAlgorithmName(params.parameters_case()).c_str()));
      }
      *use_gradient_accumulation = false;
      break;
    }
  }
  return Status::OK();
}

Status GetOptimizationAlgorithmStateVariables(
    const OptimizationParameters& params,
    std::vector<StateVariableSpecification>* state_variables) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePStpuPStpu_embedding_optimization_parameters_utilsDTcc mht_5(mht_5_v, 437, "", "./tensorflow/core/tpu/tpu_embedding_optimization_parameters_utils.cc", "GetOptimizationAlgorithmStateVariables");

  // The parameter set for the weights themselves is required to be named
  // "parameters". The rest should stay stable for compatibility. There is an
  // internal function, GetOptimizationAlgorithmStateVariableInternalIndices,
  // that needs to be updated along with this one.
  bool use_gradient_accumulation;
  TF_RETURN_IF_ERROR(
      UseGradientAccumulation(params, &use_gradient_accumulation));

  auto add_state_variable = [&](const std::string& name) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePStpuPStpu_embedding_optimization_parameters_utilsDTcc mht_6(mht_6_v, 450, "", "./tensorflow/core/tpu/tpu_embedding_optimization_parameters_utils.cc", "lambda");

    StateVariableSpecification spec;
    spec.set_name(name);
    (void)spec.mutable_user_defined();
    state_variables->push_back(spec);
  };

  switch (params.parameters_case()) {
    case OptimizationAlgorithm::kAdagrad: {
      add_state_variable("parameters");
      add_state_variable("accumulators");
      break;
    }
    case OptimizationAlgorithm::kAdagradMomentum: {
      add_state_variable("parameters");
      add_state_variable("accumulators");
      add_state_variable("momenta");
      break;
    }
    case OptimizationAlgorithm::kBoundedAdagrad: {
      add_state_variable("parameters");
      add_state_variable("accumulators");
      break;
    }
    case OptimizationAlgorithm::kStochasticGradientDescent: {
      add_state_variable("parameters");
      break;
    }
    case OptimizationAlgorithm::kFtrl: {
      add_state_variable("parameters");
      add_state_variable("accumulators");
      add_state_variable("linears");
      break;
    }
    case OptimizationAlgorithm::kAdam: {
      add_state_variable("parameters");
      add_state_variable("momenta");
      add_state_variable("velocities");
      break;
    }
    case OptimizationAlgorithm::kMomentum: {
      add_state_variable("parameters");
      add_state_variable("momenta");
      break;
    }
    case OptimizationAlgorithm::kRmsProp: {
      add_state_variable("parameters");
      add_state_variable("ms");
      add_state_variable("mom");
      break;
    }
    case OptimizationAlgorithm::kCenteredRmsProp: {
      add_state_variable("parameters");
      add_state_variable("ms");
      add_state_variable("mom");
      add_state_variable("mg");
      break;
    }
    case OptimizationAlgorithm::kMdlAdagradLight: {
      add_state_variable("parameters");
      add_state_variable("accumulators");
      add_state_variable("weights");
      add_state_variable("benefits");
      break;
    }
    case OptimizationAlgorithm::kAdadelta: {
      add_state_variable("parameters");
      add_state_variable("accumulators");
      add_state_variable("updates");
      break;
    }
    case OptimizationAlgorithm::kProximalAdagrad: {
      add_state_variable("parameters");
      add_state_variable("accumulators");
      break;
    }
    case OptimizationAlgorithm::kOnlineYogi: {
      add_state_variable("parameters");
      add_state_variable("vs");
      add_state_variable("linears");
      break;
    }
    case OptimizationAlgorithm::kProximalYogi: {
      add_state_variable("parameters");
      add_state_variable("v");
      add_state_variable("m");
      break;
    }
    case OptimizationAlgorithm::kFrequencyEstimator: {
      add_state_variable("parameters");
      add_state_variable("last_hit_step");
      break;
    }
    case OptimizationAlgorithm::kUserDefinedProgram: {
      add_state_variable("parameters");
      int num_slots = -1;
      TF_RETURN_IF_ERROR(GetBaseAuxiliaryParameterCount(params, &num_slots));
      for (int i = 0; i < num_slots; ++i) {
        add_state_variable(absl::StrCat("Slot_", i));
      }
      break;
    }
    case OptimizationAlgorithm::kAssign: {
      add_state_variable("parameters");
      break;
    }
    case OptimizationAlgorithm::PARAMETERS_NOT_SET: {
      return errors::InvalidArgument("No optimization algorithm specified");
    }
  }

  // This needs to be last for compatibility.
  if (use_gradient_accumulation) {
    StateVariableSpecification gradient_acc;
    gradient_acc.set_name("gradient_accumulators");
    gradient_acc.mutable_fill_with_constant()->set_initial_value(
        GradientAccumulatorInitialValue());
    state_variables->push_back(std::move(gradient_acc));
  }

  if (state_variables->size() > kMaxAuxiliaryParameterCount + 1) {
    return errors::InvalidArgument(
        "Optimization algorithm",
        GetOptimizationAlgorithmName(params.parameters_case()),
        "does not support gradient accumulation because it "
        "already has too many other accumulators");
  }
  return Status::OK();
}

std::vector<OptimizationAlgorithm> GetOptimizationAlgorithms() {
  return {
      OptimizationAlgorithm::kAdagrad,
      OptimizationAlgorithm::kAdagradMomentum,
      OptimizationAlgorithm::kBoundedAdagrad,
      OptimizationAlgorithm::kStochasticGradientDescent,
      OptimizationAlgorithm::kFtrl,
      OptimizationAlgorithm::kAdam,
      OptimizationAlgorithm::kMomentum,
      OptimizationAlgorithm::kRmsProp,
      OptimizationAlgorithm::kCenteredRmsProp,
      OptimizationAlgorithm::kMdlAdagradLight,
      OptimizationAlgorithm::kAdadelta,
      OptimizationAlgorithm::kProximalAdagrad,
      OptimizationAlgorithm::kOnlineYogi,
      OptimizationAlgorithm::kProximalYogi,
      OptimizationAlgorithm::kFrequencyEstimator,
      OptimizationAlgorithm::kUserDefinedProgram,
      OptimizationAlgorithm::kAssign,
  };
}

Status LoadOpShapeFunction::operator()(
    shape_inference::InferenceContext* c) const {
  int table_id;
  TF_RETURN_IF_ERROR(c->GetAttr("table_id", &table_id));
  string table_name;
  TF_RETURN_IF_ERROR(c->GetAttr("table_name", &table_name));
  // Exactly one must be non-default.
  if ((table_id >= 0) == (!table_name.empty())) {
    return errors::InvalidArgument(
        "exactly one of table_id or table_name must be non-default");
  }
  int num_shards;
  TF_RETURN_IF_ERROR(c->GetAttr("num_shards", &num_shards));
  int shard_id;
  TF_RETURN_IF_ERROR(c->GetAttr("shard_id", &shard_id));

  // Verify shapes have rank 2 and are compatible when they are
  // required to be valid.
  shape_inference::ShapeHandle parameter_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &parameter_shape));
  for (int j = 1; j < c->num_inputs(); ++j) {
    shape_inference::ShapeHandle accumulator_j_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(j), 2, &accumulator_j_shape));
    shape_inference::ShapeHandle merged;
    TF_RETURN_IF_ERROR(c->Merge(parameter_shape, accumulator_j_shape, &merged));
  }

  return Status::OK();
}

Status RetrieveOpShapeFunction::operator()(
    shape_inference::InferenceContext* c) const {
  int table_id;
  TF_RETURN_IF_ERROR(c->GetAttr("table_id", &table_id));
  string table_name;
  TF_RETURN_IF_ERROR(c->GetAttr("table_name", &table_name));
  // Exactly one must be non-default.
  if ((table_id >= 0) == (!table_name.empty())) {
    return errors::InvalidArgument(
        "exactly one of table_id or table_name must be non-default");
  }
  int num_shards;
  TF_RETURN_IF_ERROR(c->GetAttr("num_shards", &num_shards));
  int shard_id;
  TF_RETURN_IF_ERROR(c->GetAttr("shard_id", &shard_id));
  for (int j = 0; j < c->num_outputs(); ++j) {
    c->set_output(j, c->MakeShape(std::vector<shape_inference::DimensionHandle>(
                         2, c->UnknownDim())));
  }
  return Status::OK();
}

}  // namespace tpu
}  // namespace tensorflow
