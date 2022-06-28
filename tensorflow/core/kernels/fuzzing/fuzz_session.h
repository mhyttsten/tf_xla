/* Copyright 2016 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_FUZZING_FUZZ_SESSION_H_
#define TENSORFLOW_CORE_KERNELS_FUZZING_FUZZ_SESSION_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSfuzzingPSfuzz_sessionDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSfuzzingPSfuzz_sessionDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSfuzzingPSfuzz_sessionDTh() {
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


#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/public/session.h"

// Standard invoking function macro to dispatch to a fuzzer class.
#ifndef PLATFORM_WINDOWS
#define STANDARD_TF_FUZZ_FUNCTION(FuzzerClass)                              \
  extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) { \
    static FuzzerClass* fuzzer = new FuzzerClass();                         \
    return fuzzer->Fuzz(data, size);                                        \
  }
#else
// We don't compile this for Windows, MSVC doesn't like it as pywrap in Windows
// links all the code into one big object file and there are conflicting
// function names.
#define STANDARD_TF_FUZZ_FUNCTION(FuzzerClass)
#endif

// Standard builder for hooking one placeholder to one op.
#define SINGLE_INPUT_OP_BUILDER(dtype, opName)                          \
  void BuildGraph(const Scope& scope) override {                        \
    auto op_node =                                                      \
        tensorflow::ops::Placeholder(scope.WithOpName("input"), dtype); \
    (void)tensorflow::ops::opName(scope.WithOpName("output"), op_node); \
  }

namespace tensorflow {
namespace fuzzing {

// Create a TensorFlow session using a specific GraphDef created
// by BuildGraph(), and make it available for fuzzing.
// Users must override BuildGraph and FuzzImpl to specify
// (1) which operations are being fuzzed; and
// (2) How to translate the uint8_t* buffer from the fuzzer
//     to a Tensor or Tensors that are semantically appropriate
//     for the op under test.
// For the simple cases of testing a single op that takes a single
// input Tensor, use the SINGLE_INPUT_OP_BUILDER(dtype, opName) macro in place
// of defining BuildGraphDef.
//
// Typical use:
// class FooFuzzer : public FuzzSession {
//   SINGLE_INPUT_OP_BUILDER(DT_INT8, Identity);
//   void FuzzImpl(const uint8_t* data, size_t size) {
//      ... convert data and size to a Tensor, pass it to:
//      RunInputs({{"input", input_tensor}});
//
class FuzzSession {
 public:
  FuzzSession() : initialized_(false) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfuzzingPSfuzz_sessionDTh mht_0(mht_0_v, 237, "", "./tensorflow/core/kernels/fuzzing/fuzz_session.h", "FuzzSession");
}
  virtual ~FuzzSession() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfuzzingPSfuzz_sessionDTh mht_1(mht_1_v, 241, "", "./tensorflow/core/kernels/fuzzing/fuzz_session.h", "~FuzzSession");
}

  // Constructs a Graph using the supplied Scope.
  // By convention, the graph should have inputs named "input1", ...
  // "inputN", and one output node, named "output".
  // Users of FuzzSession should override this method to create their graph.
  virtual void BuildGraph(const Scope& scope) = 0;

  // Implements the logic that converts an opaque byte buffer
  // from the fuzzer to Tensor inputs to the graph.  Users must override.
  virtual void FuzzImpl(const uint8_t* data, size_t size) = 0;

  // Initializes the FuzzSession.  Not safe for multithreading.
  // Separate init function because the call to virtual BuildGraphDef
  // can't be put into the constructor.
  Status InitIfNeeded() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfuzzingPSfuzz_sessionDTh mht_2(mht_2_v, 259, "", "./tensorflow/core/kernels/fuzzing/fuzz_session.h", "InitIfNeeded");

    if (initialized_) {
      return Status::OK();
    }
    initialized_ = true;

    Scope root = Scope::DisabledShapeInferenceScope().ExitOnError();
    SessionOptions options;
    session_ = std::unique_ptr<Session>(NewSession(options));

    BuildGraph(root);

    GraphDef graph_def;
    TF_CHECK_OK(root.ToGraphDef(&graph_def));

    Status status = session_->Create(graph_def);
    if (!status.ok()) {
      // This is FATAL, because this code is designed to fuzz an op
      // within a session.  Failure to create the session means we
      // can't send any data to the op.
      LOG(FATAL) << "Could not create session: " << status.error_message();
    }
    return status;
  }

  // Runs the TF session by pulling on the "output" node, attaching
  // the supplied input_tensor to the input node(s), and discarding
  // any returned output.
  // Note: We are ignoring Status from Run here since fuzzers don't need to
  // check it (as that will slow them down and printing/logging is useless).
  void RunInputs(const std::vector<std::pair<string, Tensor> >& inputs) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfuzzingPSfuzz_sessionDTh mht_3(mht_3_v, 292, "", "./tensorflow/core/kernels/fuzzing/fuzz_session.h", "RunInputs");

    RunInputsWithStatus(inputs).IgnoreError();
  }

  // Same as RunInputs but don't ignore status
  Status RunInputsWithStatus(
      const std::vector<std::pair<string, Tensor> >& inputs) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfuzzingPSfuzz_sessionDTh mht_4(mht_4_v, 301, "", "./tensorflow/core/kernels/fuzzing/fuzz_session.h", "RunInputsWithStatus");

    return session_->Run(inputs, {}, {"output"}, nullptr);
  }

  // Dispatches to FuzzImpl;  small amount of sugar to keep the code
  // of the per-op fuzzers tiny.
  int Fuzz(const uint8_t* data, size_t size) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfuzzingPSfuzz_sessionDTh mht_5(mht_5_v, 310, "", "./tensorflow/core/kernels/fuzzing/fuzz_session.h", "Fuzz");

    Status status = InitIfNeeded();
    TF_CHECK_OK(status) << "Fuzzer graph initialization failed: "
                        << status.error_message();
    // No return value from fuzzing:  Success is defined as "did not
    // crash".  The actual application results are irrelevant.
    FuzzImpl(data, size);
    return 0;
  }

 private:
  bool initialized_;
  std::unique_ptr<Session> session_;
};

// A specialized fuzz implementation for ops that take
// a single string.  Caller must still define the op
// to plumb by overriding BuildGraph or using
// a plumbing macro.
class FuzzStringInputOp : public FuzzSession {
  void FuzzImpl(const uint8_t* data, size_t size) final {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfuzzingPSfuzz_sessionDTh mht_6(mht_6_v, 333, "", "./tensorflow/core/kernels/fuzzing/fuzz_session.h", "FuzzImpl");

    Tensor input_tensor(tensorflow::DT_STRING, TensorShape({}));
    input_tensor.scalar<tstring>()() =
        string(reinterpret_cast<const char*>(data), size);
    RunInputs({{"input", input_tensor}});
  }
};

}  // end namespace fuzzing
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_FUZZING_FUZZ_SESSION_H_
