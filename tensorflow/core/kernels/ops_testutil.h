/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_OPS_TESTUTIL_H_
#define TENSORFLOW_CORE_KERNELS_OPS_TESTUTIL_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSops_testutilDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSops_testutilDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSops_testutilDTh() {
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
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/type_index.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"

namespace tensorflow {
namespace test {

void SetOutputAttrs(OpKernelContext::Params* params,
                    std::vector<AllocatorAttributes>* attrs);

}  // namespace test

// Helpful functions to test operators.
//
// This class will eventually be replaced / heavily modified
// to use the BrainClient interface.
class OpsTestBase : public ::testing::Test {
 public:
  OpsTestBase();

  ~OpsTestBase() override;

  // Allow kernel unit tests to run on GPU
  void SetDevice(const DeviceType& device_type, std::unique_ptr<Device> device);

  void set_node_def(const NodeDef& node_def);

  // Clients can manipulate the underlying NodeDef via this accessor.
  NodeDef* node_def();

  // Initializes an operator that takes in 'input_types' as input
  // and output types as output.
  //
  // Returns the status of initialization.
  Status InitOp();

  // Only use this directly if you have a deprecated op that you need to test.
  Status InitOpWithGraphVersion(int graph_def_version);

  // Adds an input for every element described by the shape.
  // 'input_mapping' maps an index (0...NumElements(shape)) to a
  // value.
  //
  // TODO(vrv): Replace with something like a BrainClient Feed.
  template <typename T>
  void AddInput(const TensorShape& shape, std::function<T(int)> input_mapping) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSops_testutilDTh mht_0(mht_0_v, 268, "", "./tensorflow/core/kernels/ops_testutil.h", "AddInput");

    test::FillFn(AddInput(DataTypeToEnum<T>::v(), shape), input_mapping);
  }

  // Like AddInput but takes in an explicit arrayslice of data.
  template <typename T>
  void AddInputFromArray(const TensorShape& shape,
                         const gtl::ArraySlice<T> data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSops_testutilDTh mht_1(mht_1_v, 278, "", "./tensorflow/core/kernels/ops_testutil.h", "AddInputFromArray");

    test::FillValues<T>(AddInput(DataTypeToEnum<T>::v(), shape), data);
  }

  // Convenience function to add an input and populate it with the elements from
  // an initializer list converting the types as needed.
  template <typename T, typename SrcType>
  void AddInputFromList(const TensorShape& shape,
                        std::initializer_list<SrcType> data) {
    test::FillValues<T>(AddInput(DataTypeToEnum<T>::v(), shape), data);
  }

  // Adds a Resource type as input. If <container> is empty, uses the default
  // container name.
  template <typename T>
  void AddResourceInput(const string& container, const string& name,
                        T* resource) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("container: \"" + container + "\"");
   mht_2_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSops_testutilDTh mht_2(mht_2_v, 299, "", "./tensorflow/core/kernels/ops_testutil.h", "AddResourceInput");

    CHECK_GT(input_types_.size(), inputs_.size())
        << "Adding more inputs than types; perhaps you need to call MakeOp";
    ResourceMgr* rm = device_->resource_manager();
    std::string container_name =
        container.empty() ? rm->default_container() : container;
    EXPECT_TRUE(rm->Create(container_name, name, resource).ok());
    AddResourceInputInternal(container_name, name, TypeIndex::Make<T>());
  }

  // Runs an operation producing 'num_outputs' outputs.
  //
  // Returns the context's status after running the operation.
  Status RunOpKernel();

  // Returns the tensor input for 'input_index'.
  //
  // REQUIRES: 0 <= input_index < context_->num_inputs()
  const Tensor& GetInput(int input_index) const;

  TensorValue mutable_input(int input_index);

  // Returns the tensor output for 'output_index'.
  //
  // REQUIRES: 0 <= output_index < context_->num_outputs()
  Tensor* GetOutput(int output_index);

  Allocator* allocator();

  OpKernel* op_kernel();

  const DataTypeVector& output_types() const;

 protected:
  Tensor* AddInput(DataType dtype, const TensorShape& shape);
  void AddResourceInputInternal(const std::string& container_name,
                                const std::string& name,
                                const TypeIndex& type_index);

  // device_mgr_ owns device_.
  std::unique_ptr<DeviceMgr> device_mgr_;
  Device* device_;

  // The device allocator, or the managed_allocator_ below if running on GPU.
  Allocator* allocator_;

  std::unique_ptr<OpKernel> kernel_;
  std::unique_ptr<ScopedStepContainer> step_container_;
  NodeDef node_def_;
  DataTypeVector input_types_;
  DeviceType device_type_;

  mutex lock_for_refs_;  // Used as the Mutex for inputs added as refs

  gtl::InlinedVector<TensorValue, 4> inputs_;
  // Owns Tensors.
  std::vector<Tensor*> tensors_;
  // Copies of the outputs in unified memory (host and device accessible).
  std::vector<Tensor*> managed_outputs_;

  std::unique_ptr<OpKernelContext::Params> params_;
  std::unique_ptr<OpKernelContext> context_;
  // Unified memory allocator, only used when running on GPU.
  std::unique_ptr<Allocator> managed_allocator_;

  std::unique_ptr<FunctionLibraryDefinition> flib_def_;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr_;
  std::unique_ptr<thread::ThreadPool> thread_pool_;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(OpsTestBase);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_OPS_TESTUTIL_H_
