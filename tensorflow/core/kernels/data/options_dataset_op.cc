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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSoptions_dataset_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSoptions_dataset_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSoptions_dataset_opDTcc() {
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
==============================================================================*/
#include "tensorflow/core/kernels/data/options_dataset_op.h"

#include "absl/memory/memory.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/dataset_options.pb.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace tensorflow {
namespace data {

/* static */ constexpr const char* const OptionsDatasetOp::kDatasetType;
/* static */ constexpr const char* const OptionsDatasetOp::kInputDataset;
/* static */ constexpr const char* const OptionsDatasetOp::kOutputTypes;
/* static */ constexpr const char* const OptionsDatasetOp::kOutputShapes;
/* static */ constexpr const char* const OptionsDatasetOp::kSerializedOptions;

class OptionsDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input,
          const string& serialized_options)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        serialized_options_(serialized_options) {
    input_->Ref();
    Options options;
    OP_REQUIRES(ctx, options.ParseFromString(serialized_options),
                errors::InvalidArgument(absl::StrCat(
                    "Could not parse ", OptionsDatasetOp::kSerializedOptions,
                    " as valid Options.")));
    set_options(options);
  }

  ~Dataset() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSoptions_dataset_opDTcc mht_0(mht_0_v, 219, "", "./tensorflow/core/kernels/data/options_dataset_op.cc", "~Dataset");
 input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    DCHECK(false) << "OptionsDatasetOp::Dataset::MakeIteratorInternal is not "
                     "expected to be called because it is supposed to forward "
                     "the iterator to its input dataset(s).";
    LOG(ERROR) << "Datasets of type " << type_string()
               << " forwards its iterator to its input dataset. "
                  "`MakeIteratorInternal` is not implemented.";
    return nullptr;
  }

  const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSoptions_dataset_opDTcc mht_1(mht_1_v, 235, "", "./tensorflow/core/kernels/data/options_dataset_op.cc", "output_dtypes");

    return input_->output_dtypes();
  }
  const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSoptions_dataset_opDTcc mht_2(mht_2_v, 241, "", "./tensorflow/core/kernels/data/options_dataset_op.cc", "output_shapes");

    return input_->output_shapes();
  }

  int64_t CardinalityInternal() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSoptions_dataset_opDTcc mht_3(mht_3_v, 248, "", "./tensorflow/core/kernels/data/options_dataset_op.cc", "CardinalityInternal");
 return input_->Cardinality(); }

  Status Get(OpKernelContext* ctx, int64 index,
             std::vector<Tensor>* out_tensors) const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSoptions_dataset_opDTcc mht_4(mht_4_v, 254, "", "./tensorflow/core/kernels/data/options_dataset_op.cc", "Get");

    TF_RETURN_IF_ERROR(CheckRandomAccessCompatible(index));
    return input_->Get(ctx, index, out_tensors);
  }

  string DebugString() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSoptions_dataset_opDTcc mht_5(mht_5_v, 262, "", "./tensorflow/core/kernels/data/options_dataset_op.cc", "DebugString");

    return name_utils::DatasetDebugString(kDatasetType);
  }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSoptions_dataset_opDTcc mht_6(mht_6_v, 269, "", "./tensorflow/core/kernels/data/options_dataset_op.cc", "InputDatasets");

    inputs->push_back(input_);
    return Status::OK();
  }

  Status CheckExternalState() const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSoptions_dataset_opDTcc mht_7(mht_7_v, 277, "", "./tensorflow/core/kernels/data/options_dataset_op.cc", "CheckExternalState");

    return input_->CheckExternalState();
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSoptions_dataset_opDTcc mht_8(mht_8_v, 287, "", "./tensorflow/core/kernels/data/options_dataset_op.cc", "AsGraphDefInternal");

    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    AttrValue serialized_options_attr;
    b->BuildAttrValue(serialized_options_, &serialized_options_attr);
    TF_RETURN_IF_ERROR(b->AddDataset(
        this, {input_graph_node},
        {std::make_pair(kSerializedOptions, serialized_options_attr)}, output));
    return Status::OK();
  }

 private:
  const DatasetBase* input_;
  const tstring serialized_options_;
};

void OptionsDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase** output) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSoptions_dataset_opDTcc mht_9(mht_9_v, 306, "", "./tensorflow/core/kernels/data/options_dataset_op.cc", "OptionsDatasetOp::MakeDataset");

  DatasetBase* input;
  OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(0), &input));
  *output = new Dataset(ctx, input, serialized_options_);
}

OptionsDatasetOp::OptionsDatasetOp(OpKernelConstruction* ctx)
    : DatasetOpKernel(ctx) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSoptions_dataset_opDTcc mht_10(mht_10_v, 316, "", "./tensorflow/core/kernels/data/options_dataset_op.cc", "OptionsDatasetOp::OptionsDatasetOp");

  OP_REQUIRES_OK(ctx, ctx->GetAttr(kSerializedOptions, &serialized_options_));
}

namespace {
REGISTER_KERNEL_BUILDER(Name("OptionsDataset").Device(DEVICE_CPU).Priority(2),
                        OptionsDatasetOp);
REGISTER_KERNEL_BUILDER(Name("OptionsDataset")
                            .Device(DEVICE_GPU)
                            .HostMemory("input_dataset")
                            .HostMemory("handle")
                            .Priority(1),
                        OptionsDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
