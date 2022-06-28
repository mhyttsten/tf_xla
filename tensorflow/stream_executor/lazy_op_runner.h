/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
------------------------------------------------------------------------------*/

#ifndef TENSORFLOW_STREAM_EXECUTOR_LAZY_OP_RUNNER_H_
#define TENSORFLOW_STREAM_EXECUTOR_LAZY_OP_RUNNER_H_
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
class MHTracer_DTPStensorflowPSstream_executorPSlazy_op_runnerDTh {
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
   MHTracer_DTPStensorflowPSstream_executorPSlazy_op_runnerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPSlazy_op_runnerDTh() {
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


#include "tensorflow/stream_executor/dnn.h"
#include "tensorflow/stream_executor/stream.h"
// #include "tensorflow/stream_executor/stream_executor_pimpl.h"

namespace stream_executor {
namespace dnn {

// A lazily-initialized OpRunner from an AlgorithmDesc.
//
// This exists to hold a choice of conv algorithm for a particular config,
// initialize its OpRunner at most once, and defer that initialization until the
// config is first needed.  This allows AoT autotuning to load configurations
// for all convolutions it knows about, without doing expensive initialization
// (e.g. runtime codegen) and retaining non-negligible resources (e.g.  compiled
// kernels) for potentially irrelevant configurations.  It also enables XLA conv
// thunks to defer binding to a particular stream executor until the first run.
//
// `Op` must satisfy the following "concept":
//
// struct Op {
//   // The function type signature parameter of an OpRunner.
//   using Signature = _;
//
//   // The parameter to be used by GetOrCreateRunner.
//   struct Config;
//
//   // Use a StreamExecutor to create an OpRunner.
//   static StatusOr<OpRunner<Config>> OpRunnerFromDesc(
//       const AlgorithmDesc& desc, Config config, StreamExecutor* stream);
// };
template <typename Op>
class LazyOpRunner {
 public:
  // Construct from a pre-initialized OpRunner; all calls to GetOrCreateRunner
  // will return a pointer to exactly this runner.
  static port::StatusOr<std::unique_ptr<LazyOpRunner>> FromOpRunner(
      std::unique_ptr<const OpRunner<typename Op::Signature>> runner) {
    if (!runner) {
      return port::InternalError("Null runner argument to FromOpRunner");
    }
    SE_ASSIGN_OR_RETURN(auto desc, runner->ToAlgorithmDesc());
    // Private constructor cannot be called by make_unique :(
    return {std::unique_ptr<LazyOpRunner>(
        new LazyOpRunner(desc, std::move(runner)))};
  }

  // Construct from an AlgorithmDesc, with no pre-initialized OpRunner; it will
  // be created on the first call to GetOrCreateRunner.
  explicit LazyOpRunner(AlgorithmDesc desc) : LazyOpRunner(desc, nullptr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPSlazy_op_runnerDTh mht_0(mht_0_v, 236, "", "./tensorflow/stream_executor/lazy_op_runner.h", "LazyOpRunner");
}

  // Returns an already-initialized OpRunner if available, or creates one.
  //
  // Invariant: a particular instance of this class shall only receive calls
  // with identical `config`s and `stream_executor`s.  If the config is changed,
  // only the first config it sees will have any effect, and second and
  // subsequent configs will be ignored.  If the stream executor is changed,
  // some operations on the returned `OpRunner` using the changed stream
  // executor will be errors.
  //
  // The result is owned by LazyOpRunner.
  port::StatusOr<const OpRunner<typename Op::Signature>*> GetOrCreateRunner(
      typename Op::Config config, Stream* stream) {
    absl::MutexLock lock(&mu_);
    if (!runner_) {
      SE_ASSIGN_OR_RETURN(runner_, Op::RunnerFromAlgorithmDesc(
                                       desc_, std::move(config), stream));
    }
    return runner_.get();
  }

  // Get the contained runner with the invariant that it's already initialized.
  port::StatusOr<const OpRunner<typename Op::Signature>*> GetRunner() {
    absl::MutexLock lock(&mu_);
    if (!runner_) {
      return port::InternalError("LazyOpRunner::GetRunner: not initialized");
    }
    return runner_.get();
  }

  bool operator==(const LazyOpRunner& other) const {
    return desc_ == other.desc_;
  }

  std::string ToString() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPSlazy_op_runnerDTh mht_1(mht_1_v, 274, "", "./tensorflow/stream_executor/lazy_op_runner.h", "ToString");
 return desc_.ToString(); }

  const AlgorithmDesc& ToAlgorithmDesc() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPSlazy_op_runnerDTh mht_2(mht_2_v, 279, "", "./tensorflow/stream_executor/lazy_op_runner.h", "ToAlgorithmDesc");
 return desc_; }

 private:
  LazyOpRunner(AlgorithmDesc desc,
               std::unique_ptr<const OpRunner<typename Op::Signature>> runner)
      : desc_(std::move(desc)), runner_(std::move(runner)) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPSlazy_op_runnerDTh mht_3(mht_3_v, 287, "", "./tensorflow/stream_executor/lazy_op_runner.h", "LazyOpRunner");
}

  AlgorithmDesc desc_;
  absl::Mutex mu_;
  std::unique_ptr<const OpRunner<typename Op::Signature>> runner_
      ABSL_GUARDED_BY(mu_);
};

// Implementation of the concept required by LazyOpRunner, for ConvRunner.
struct ConvOp {
  using Signature = ConvSignature;

  struct Config {
    ConvolutionKind kind;
    DataType input_type, output_type;
    const BatchDescriptor& input_descriptor;
    const FilterDescriptor& filter_descriptor;
    const BatchDescriptor& output_descriptor;
    const ConvolutionDescriptor& convolution_descriptor;
  };

  static port::StatusOr<std::unique_ptr<const OpRunner<ConvSignature>>>
  RunnerFromAlgorithmDesc(const AlgorithmDesc& desc, Config config,
                          Stream* stream) {
    return stream->ConvolveRunnerFromDesc(
        desc, config.kind, config.input_type, config.output_type,
        config.input_descriptor, config.filter_descriptor,
        config.output_descriptor, config.convolution_descriptor);
  }
};

// Implementation of the concept required by LazyOpRunner, for LazyConvRunner.
struct FusedConvOp {
  using Signature = FusedConvSignature;

  struct Config {
    ConvolutionKind kind;
    DataType input_type, bias_type, output_type;
    double conv_scale, side_input_scale;
    const BatchDescriptor& input_descriptor;
    const FilterDescriptor& filter_descriptor;
    const BatchDescriptor& bias_descriptor;
    const BatchDescriptor& output_descriptor;
    const ConvolutionDescriptor& convolution_descriptor;
    ActivationMode activation_mode;
  };

  static port::StatusOr<std::unique_ptr<const OpRunner<FusedConvSignature>>>
  RunnerFromAlgorithmDesc(const AlgorithmDesc& desc, Config config,
                          Stream* stream) {
    return stream->FusedConvolveRunnerFromDesc(
        desc, config.kind, config.input_type, config.bias_type,
        config.output_type, config.conv_scale, config.side_input_scale,
        config.input_descriptor, config.filter_descriptor,
        config.bias_descriptor, config.output_descriptor,
        config.convolution_descriptor, config.activation_mode);
  }
};

}  // namespace dnn
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_LAZY_OP_RUNNER_H_
