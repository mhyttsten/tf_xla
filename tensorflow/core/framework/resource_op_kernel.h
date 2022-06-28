/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_FRAMEWORK_RESOURCE_OP_KERNEL_H_
#define TENSORFLOW_CORE_FRAMEWORK_RESOURCE_OP_KERNEL_H_
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
class MHTracer_DTPStensorflowPScorePSframeworkPSresource_op_kernelDTh {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_op_kernelDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSresource_op_kernelDTh() {
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


#include <string>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// ResourceOpKernel<T> is a virtual base class for resource op implementing
// interface type T. The inherited op looks up the resource name (determined by
// ContainerInfo), and creates a new resource if necessary.
//
// Requirements:
//  - Op must be marked as stateful.
//  - Op must have `container` and `shared_name` attributes. Empty `container`
//  means using the default container. Empty `shared_name` means private
//  resource.
//  - Subclass must override CreateResource().
//  - Subclass is encouraged to override VerifyResource().
template <typename T>
class ResourceOpKernel : public OpKernel {
 public:
  explicit ResourceOpKernel(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_op_kernelDTh mht_0(mht_0_v, 214, "", "./tensorflow/core/framework/resource_op_kernel.h", "ResourceOpKernel");

    has_resource_type_ = (context->output_type(0) == DT_RESOURCE);
    if (!has_resource_type_) {
      // The resource variant of the op may be placed on non-CPU devices, but
      // this allocation is always on the host. Fortunately we don't need it in
      // the resource case.
      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DT_STRING, TensorShape({2}), &tensor_));
    }
  }

  // The resource is deleted from the resource manager only when it is private
  // to kernel. Ideally the resource should be deleted when it is no longer held
  // by anyone, but it would break backward compatibility.
  ~ResourceOpKernel() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_op_kernelDTh mht_1(mht_1_v, 231, "", "./tensorflow/core/framework/resource_op_kernel.h", "~ResourceOpKernel");

    if (resource_ != nullptr) {
      resource_->Unref();
      if (cinfo_.resource_is_private_to_kernel()) {
        if (!cinfo_.resource_manager()
                 ->template Delete<T>(cinfo_.container(), cinfo_.name())
                 .ok()) {
          // Do nothing; the resource can have been deleted by session resets.
        }
      }
    }
  }

  void Compute(OpKernelContext* context) override TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    if (resource_ == nullptr) {
      ResourceMgr* mgr = context->resource_manager();
      OP_REQUIRES_OK(context, cinfo_.Init(mgr, def()));

      T* resource;
      OP_REQUIRES_OK(context,
                     mgr->LookupOrCreate<T>(
                         cinfo_.container(), cinfo_.name(), &resource,
                         [this](T** ret) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
                           Status s = CreateResource(ret);
                           if (!s.ok() && *ret != nullptr) {
                             CHECK((*ret)->Unref());
                           }
                           return s;
                         }));

      Status s = VerifyResource(resource);
      if (TF_PREDICT_FALSE(!s.ok())) {
        resource->Unref();
        context->SetStatus(s);
        return;
      }

      if (!has_resource_type_) {
        auto h = tensor_.template flat<tstring>();
        h(0) = cinfo_.container();
        h(1) = cinfo_.name();
      }
      resource_ = resource;
    }
    if (has_resource_type_) {
      OP_REQUIRES_OK(context, MakeResourceHandleToOutput(
                                  context, 0, cinfo_.container(), cinfo_.name(),
                                  TypeIndex::Make<T>()));
    } else {
      context->set_output_ref(0, &mu_, &tensor_);
    }
  }

 protected:
  // Variables accessible from subclasses.
  mutex mu_;
  ContainerInfo cinfo_ TF_GUARDED_BY(mu_);
  T* resource_ TF_GUARDED_BY(mu_) = nullptr;

 private:
  // Must return a T descendant allocated with new that ResourceOpKernel will
  // take ownership of.
  virtual Status CreateResource(T** resource)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) = 0;

  // During the first Compute(), resource is either created or looked up using
  // shared_name. In the latter case, the resource found should be verified if
  // it is compatible with this op's configuration. The verification may fail in
  // cases such as two graphs asking queues of the same shared name to have
  // inconsistent capacities.
  virtual Status VerifyResource(T* resource) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSresource_op_kernelDTh mht_2(mht_2_v, 305, "", "./tensorflow/core/framework/resource_op_kernel.h", "VerifyResource");
 return Status::OK(); }

  Tensor tensor_ TF_GUARDED_BY(mu_);

  // Is the output of the operator of type DT_RESOURCE?
  bool has_resource_type_;
};
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_RESOURCE_OP_KERNEL_H_
