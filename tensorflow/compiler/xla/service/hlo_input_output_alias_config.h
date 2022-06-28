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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_INPUT_OUTPUT_ALIAS_CONFIG_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_INPUT_OUTPUT_ALIAS_CONFIG_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_input_output_alias_configDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_input_output_alias_configDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_input_output_alias_configDTh() {
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


#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {

class HloModule;

// This class specifies the alias map from output index to parameter number and
// parameter index in the entry computation.
class HloInputOutputAliasConfig {
 public:
  // The kind of aliases which can be set. A kMayAlias is one setup at
  // compilation time by the user, and has to be respected. A kMustAlias one
  // might be setup by the compiler, if it decides it is convenient to do so.
  enum AliasKind {
    kMayAlias,
    kMustAlias,
  };
  // Defines the alias information for a given output buffer. A given output
  // buffer shape index can refer only to one parameter+index.
  struct Alias {
    Alias(int64_t parameter_number, ShapeIndex parameter_index,
          AliasKind kind = kMayAlias)
        : parameter_number(parameter_number),
          parameter_index(std::move(parameter_index)),
          kind(kind) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_input_output_alias_configDTh mht_0(mht_0_v, 218, "", "./tensorflow/compiler/xla/service/hlo_input_output_alias_config.h", "Alias");
}

    int64_t parameter_number;
    ShapeIndex parameter_index;
    AliasKind kind;

    bool must_alias() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_input_output_alias_configDTh mht_1(mht_1_v, 227, "", "./tensorflow/compiler/xla/service/hlo_input_output_alias_config.h", "must_alias");
 return kind == kMustAlias; }

    std::string ToString() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_input_output_alias_configDTh mht_2(mht_2_v, 232, "", "./tensorflow/compiler/xla/service/hlo_input_output_alias_config.h", "ToString");

      return absl::StrFormat("(%lld, %s, %s)", parameter_number,
                             parameter_index.ToString(),
                             kind == kMustAlias ? "must-alias" : "may-alias");
    }
  };

  HloInputOutputAliasConfig() = default;

  explicit HloInputOutputAliasConfig(Shape output_shape)
      : alias_(std::move(output_shape)) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_input_output_alias_configDTh mht_3(mht_3_v, 245, "", "./tensorflow/compiler/xla/service/hlo_input_output_alias_config.h", "HloInputOutputAliasConfig");
}

  virtual ~HloInputOutputAliasConfig() = default;

  // Sets up alias config from `output_index` to `param_index` at
  // `param_number`.
  Status SetUpAlias(const ShapeIndex& output_index, int64_t param_number,
                    const ShapeIndex& param_index,
                    AliasKind must_alias = kMayAlias);

  // Returns true if the given parameter is aliased with one of the output
  // buffers.
  bool ParameterHasAlias(int64_t param_number,
                         const ShapeIndex& param_index) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_input_output_alias_configDTh mht_4(mht_4_v, 261, "", "./tensorflow/compiler/xla/service/hlo_input_output_alias_config.h", "ParameterHasAlias");

    return GetAliasedOutput(param_number, param_index).has_value();
  }

  // Checks whether the provided output index has already been aliased.
  bool OutputHasAlias(const ShapeIndex& output_index) const;

  // (De)Serializes an HloInputOutputAliasConfig to/from an
  // HloInputOutputAliasProto.
  HloInputOutputAliasProto ToProto() const;

  static StatusOr<HloInputOutputAliasConfig> CreateFromProto(
      Shape output_shape, const HloInputOutputAliasProto& proto);

  // Returns the output index that the given parameter and parameter index is
  // aliased with. A nullopt is returned if there is no output that is aliased
  // with the parameter number and index.
  absl::optional<ShapeIndex> GetAliasedOutput(
      int64_t param_number, const ShapeIndex& param_index) const;

  // Returns the number of parameter and index of the parameter buffer that the
  // given output buffer index is aliased with. A nullopt is returned if there
  // is no parameter is aliased with the specific output.
  absl::optional<Alias> GetAliasedParameter(
      const ShapeIndex& output_index) const;

  // Returns if the parameter at the given parameter number and parameter
  // index must-alias with an output.
  bool ParameterMustAlias(int64_t param_number,
                          const ShapeIndex& param_index) const;

  using AliasFn =
      std::function<void(const ShapeIndex& output_index, const Alias&)>;

  // Iterates through each aliased output and input.
  void ForEachAlias(AliasFn fn) const;

  using AliasFnWithStatus =
      std::function<Status(const ShapeIndex& output_index, const Alias&)>;

  // Verifies that the given config is valid for the given module.
  // Specifically, the config's input and output should be in-bound and size of
  // the aliased buffers should match.
  Status Verify(const HloModule& module,
                std::function<int64_t(const Shape&)> size_func_) const;

  Status ForEachAliasWithStatus(AliasFnWithStatus fn) const;

  // Returns the shape of the output of the alias config.
  const Shape& shape() const;

  std::string ToString() const;

  std::string ToShortString() const;

 private:
  // A ShapeTree which indicates the list of buffers that's expected to be
  // aliased. The key on this shape tree represents the output index. The value
  // is an Alias data structure which defines the input parameter coordinates.
  // If the value is nullopt, it means there is no parameter aliasing for this
  // output.
  ShapeTree<absl::optional<Alias>> alias_;
};

std::ostream& operator<<(std::ostream& out,
                         const HloInputOutputAliasConfig& config);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_INPUT_OUTPUT_ALIAS_CONFIG_H_
