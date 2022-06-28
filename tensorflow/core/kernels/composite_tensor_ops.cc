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
class MHTracer_DTPStensorflowPScorePSkernelsPScomposite_tensor_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPScomposite_tensor_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPScomposite_tensor_opsDTcc() {
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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/kernels/composite_tensor_variant.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/protobuf/composite_tensor_variant.pb.h"
#include "tensorflow/core/protobuf/struct.pb.h"

namespace tensorflow {

class CompositeTensorVariantFromComponents : public OpKernel {
 public:
  explicit CompositeTensorVariantFromComponents(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScomposite_tensor_opsDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/kernels/composite_tensor_ops.cc", "CompositeTensorVariantFromComponents");

    string type_spec_string;
    OP_REQUIRES_OK(context, context->GetAttr("metadata", &type_spec_string));
    OP_REQUIRES(context, metadata_.ParseFromString(type_spec_string),
                errors::InvalidArgument("Error parsing metadata"));
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScomposite_tensor_opsDTcc mht_1(mht_1_v, 209, "", "./tensorflow/core/kernels/composite_tensor_ops.cc", "Compute");

    OpInputList components_in;
    OP_REQUIRES_OK(context, context->input_list("components", &components_in));

    Tensor* encoded;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}), &encoded));

    std::vector<Tensor> components{components_in.begin(), components_in.end()};
    encoded->flat<Variant>()(0) =
        CompositeTensorVariant(metadata_, absl::MakeSpan(components));
  }

 private:
  CompositeTensorVariantMetadata metadata_;
};

class CompositeTensorVariantToComponents : public OpKernel {
 public:
  explicit CompositeTensorVariantToComponents(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScomposite_tensor_opsDTcc mht_2(mht_2_v, 232, "", "./tensorflow/core/kernels/composite_tensor_ops.cc", "CompositeTensorVariantToComponents");

    string type_spec_string;
    OP_REQUIRES_OK(context, context->GetAttr("metadata", &type_spec_string));
    OP_REQUIRES(context, metadata_.ParseFromString(type_spec_string),
                errors::InvalidArgument("Error parsing `metadata`"));

    OP_REQUIRES_OK(context,
                   context->GetAttr("Tcomponents", &component_dtypes_));
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScomposite_tensor_opsDTcc mht_3(mht_3_v, 245, "", "./tensorflow/core/kernels/composite_tensor_ops.cc", "Compute");

    Tensor encoded_t = context->input(0);
    auto* encoded = encoded_t.flat<Variant>()(0).get<CompositeTensorVariant>();

    // Check that the encoded TypeSpec is compatible with the expected TypeSpec.
    // For now, we just check that the class matches.
    //
    // TODO(b/173744905): Update this to do a generic compatibility check. This
    // would require replacing the current design, where Python subclasses of
    // TypeSpec can override is_compatible, with a design where compatibility
    // can be deterministically determined from the metadata.
    auto expected_class = metadata_.type_spec_proto().type_spec_class();
    auto actual_class = encoded->metadata().type_spec_proto().type_spec_class();
    OP_REQUIRES(
        context, expected_class == actual_class,
        errors::InvalidArgument(
            "Expected a ", TypeSpecProto::TypeSpecClass_Name(expected_class),
            " (based on `type_spec`), but `encoded` contains a ",
            TypeSpecProto::TypeSpecClass_Name(actual_class)));

    // Extract the component tensors.
    OpOutputList components;
    OP_REQUIRES_OK(context, context->output_list("components", &components));
    int num_components = encoded->flat_components().size();

    OP_REQUIRES(context, component_dtypes_.size() == num_components,
                errors::InvalidArgument("Encoded value has ", num_components,
                                        " tensor components; expected ",
                                        component_dtypes_.size(),
                                        " components based on type_spec"));

    for (int i = 0; i < component_dtypes_.size(); i++) {
      const Tensor& component = encoded->flat_components()[i];
      OP_REQUIRES(context, component_dtypes_[i] == component.dtype(),
                  errors::InvalidArgument("Tensor component ", i, " had dtype ",
                                          DataType_Name(component.dtype()),
                                          "; expected dtype ",
                                          DataType_Name(component_dtypes_[i])));
      components.set(i, component);
    }
  }

 private:
  CompositeTensorVariantMetadata metadata_;
  std::vector<DataType> component_dtypes_;
};

REGISTER_KERNEL_BUILDER(
    Name("CompositeTensorVariantToComponents").Device(DEVICE_CPU),
    CompositeTensorVariantToComponents);
REGISTER_KERNEL_BUILDER(
    Name("CompositeTensorVariantFromComponents").Device(DEVICE_CPU),
    CompositeTensorVariantFromComponents);

}  // namespace tensorflow
