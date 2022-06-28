/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// This file has utilities that facilitates creating new delegates.
// - SimpleDelegateKernelInterface: Represents a Kernel which handles a subgraph
// to be delegated. It has Init/Prepare/Invoke which are going to be called
// during inference, similar to TFLite Kernels. Delegate owner should implement
// this interface to build/prepare/invoke the delegated subgraph.
// - SimpleDelegateInterface:
// This class wraps TFLiteDelegate and users need to implement the interface and
// then call TfLiteDelegateFactory::CreateSimpleDelegate(...) to get
// TfLiteDelegate* that can be passed to ModifyGraphWithDelegate and free it via
// TfLiteDelegateFactory::DeleteSimpleDelegate(...).
// or call TfLiteDelegateFactory::Create(...) to get a std::unique_ptr
// TfLiteDelegate that can also be passed to ModifyGraphWithDelegate, in which
// case TfLite interpereter takes the memory ownership of the delegate.
#ifndef TENSORFLOW_LITE_DELEGATES_UTILS_SIMPLE_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_UTILS_SIMPLE_DELEGATE_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsPSsimple_delegateDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsPSsimple_delegateDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsPSsimple_delegateDTh() {
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

#include "tensorflow/lite/c/common.h"

namespace tflite {

using TfLiteDelegateUniquePtr =
    std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>;

// Users should inherit from this class and implement the interface below.
// Each instance represents a single part of the graph (subgraph).
class SimpleDelegateKernelInterface {
 public:
  virtual ~SimpleDelegateKernelInterface() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsPSsimple_delegateDTh mht_0(mht_0_v, 214, "", "./tensorflow/lite/delegates/utils/simple_delegate.h", "~SimpleDelegateKernelInterface");
}

  // Initializes a delegated subgraph.
  // The nodes in the subgraph are inside TfLiteDelegateParams->nodes_to_replace
  virtual TfLiteStatus Init(TfLiteContext* context,
                            const TfLiteDelegateParams* params) = 0;

  // Will be called by the framework. Should handle any needed preparation
  // for the subgraph e.g. allocating buffers, compiling model.
  // Returns status, and signalling any errors.
  virtual TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) = 0;

  // Actual subgraph inference should happen on this call.
  // Returns status, and signalling any errors.
  // NOTE: Tensor data pointers (tensor->data) can change every inference, so
  // the implementation of this method needs to take that into account.
  virtual TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) = 0;
};

// Pure Interface that clients should implement.
// The Interface represents a delegate capabilities and provide factory
// for SimpleDelegateKernelInterface
//
// Clients should implement the following methods:
// - IsNodeSupportedByDelegate
// - Initialize
// - name
// - CreateDelegateKernelInterface
class SimpleDelegateInterface {
 public:
  // Options for configuring a delegate.
  struct Options {
    // Maximum number of delegated subgraph, values <=0 means unlimited.
    int max_delegated_partitions = 0;

    // The minimum number of nodes allowed in a delegated graph, values <=0
    // means unlimited.
    int min_nodes_per_partition = 0;
  };

  virtual ~SimpleDelegateInterface() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsPSsimple_delegateDTh mht_1(mht_1_v, 257, "", "./tensorflow/lite/delegates/utils/simple_delegate.h", "~SimpleDelegateInterface");
}

  // Returns true if 'node' is supported by the delegate. False otherwise.
  virtual bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
                                         const TfLiteNode* node,
                                         TfLiteContext* context) const = 0;

  // Initialize the delegate before finding and replacing TfLite nodes with
  // delegate kernels, for example, retrieving some TFLite settings from
  // 'context'.
  virtual TfLiteStatus Initialize(TfLiteContext* context) = 0;

  // Returns a name that identifies the delegate.
  // This name is used for debugging/logging/profiling.
  virtual const char* Name() const = 0;

  // Returns instance of an object that implements the interface
  // SimpleDelegateKernelInterface.
  // An instance of SimpleDelegateKernelInterface represents one subgraph to
  // be delegated.
  // Caller takes ownership of the returned object.
  virtual std::unique_ptr<SimpleDelegateKernelInterface>
  CreateDelegateKernelInterface() = 0;

  // Returns SimpleDelegateInterface::Options which has the delegate options.
  virtual SimpleDelegateInterface::Options DelegateOptions() const = 0;
};

// Factory class that provides static methods to deal with SimpleDelegate
// creation and deletion.
class TfLiteDelegateFactory {
 public:
  // Creates TfLiteDelegate from the provided SimpleDelegateInterface.
  // The returned TfLiteDelegate should be deleted using DeleteSimpleDelegate.
  // A simple usage of the flags bit mask:
  // CreateSimpleDelegate(..., kTfLiteDelegateFlagsAllowDynamicTensors |
  // kTfLiteDelegateFlagsRequirePropagatedShapes)
  static TfLiteDelegate* CreateSimpleDelegate(
      std::unique_ptr<SimpleDelegateInterface> simple_delegate,
      int64_t flags = kTfLiteDelegateFlagsNone);

  // Deletes 'delegate' the passed pointer must be the one returned
  // from CreateSimpleDelegate.
  // This function will destruct the SimpleDelegate object too.
  static void DeleteSimpleDelegate(TfLiteDelegate* delegate);

  // A convenient function wrapping the above two functions and returning a
  // std::unique_ptr type for auto memory management.
  inline static TfLiteDelegateUniquePtr Create(
      std::unique_ptr<SimpleDelegateInterface> simple_delegate) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsPSsimple_delegateDTh mht_2(mht_2_v, 309, "", "./tensorflow/lite/delegates/utils/simple_delegate.h", "Create");

    return TfLiteDelegateUniquePtr(
        CreateSimpleDelegate(std::move(simple_delegate)), DeleteSimpleDelegate);
  }
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_UTILS_SIMPLE_DELEGATE_H_
