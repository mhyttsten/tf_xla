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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_datasetDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_datasetDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_datasetDTcc() {
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
#include "tensorflow/core/kernels/data/window_dataset.h"

#include <string>
#include <utility>

#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kInputs[] = "inputs";
constexpr char kOutputTypes[] = "output_types";
constexpr char kOutputShapes[] = "output_shapes";
constexpr char kWindow[] = "Window";
constexpr char kWindowOp[] = "WindowOp";
constexpr char kCurIndex[] = "i";

class Window : public DatasetBase {
 public:
  Window(std::vector<std::vector<Tensor>> elements, DataTypeVector output_types,
         std::vector<PartialTensorShape> output_shapes)
      : DatasetBase(DatasetContext({kWindowOp, kWindow})),
        elements_(std::move(elements)),
        output_types_(std::move(output_types)),
        output_shapes_(std::move(output_shapes)) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_datasetDTcc mht_0(mht_0_v, 212, "", "./tensorflow/core/kernels/data/window_dataset.cc", "Window");
}

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(
        Iterator::Params{this, name_utils::IteratorPrefix(kWindow, prefix)});
  }

  const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_datasetDTcc mht_1(mht_1_v, 223, "", "./tensorflow/core/kernels/data/window_dataset.cc", "output_dtypes");
 return output_types_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_datasetDTcc mht_2(mht_2_v, 228, "", "./tensorflow/core/kernels/data/window_dataset.cc", "output_shapes");

    return output_shapes_;
  }

  int64_t AllocatedBytes() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_datasetDTcc mht_3(mht_3_v, 235, "", "./tensorflow/core/kernels/data/window_dataset.cc", "AllocatedBytes");

    int64_t allocated_bytes = 0;
    for (auto& element : elements_) {
      allocated_bytes += GetAllocatedBytes(element);
    }
    return allocated_bytes;
  }

  int64_t TotalBytes() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_datasetDTcc mht_4(mht_4_v, 246, "", "./tensorflow/core/kernels/data/window_dataset.cc", "TotalBytes");

    int64_t total_bytes = 0;
    for (auto& element : elements_) {
      total_bytes += GetTotalBytes(element);
    }
    return total_bytes;
  }

  int64_t CardinalityInternal() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_datasetDTcc mht_5(mht_5_v, 257, "", "./tensorflow/core/kernels/data/window_dataset.cc", "CardinalityInternal");
 return elements_.size(); }

  string DebugString() const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_datasetDTcc mht_6(mht_6_v, 262, "", "./tensorflow/core/kernels/data/window_dataset.cc", "DebugString");
 return kWindow; }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_datasetDTcc mht_7(mht_7_v, 267, "", "./tensorflow/core/kernels/data/window_dataset.cc", "InputDatasets");

    return Status::OK();
  }

  Status CheckExternalState() const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_datasetDTcc mht_8(mht_8_v, 274, "", "./tensorflow/core/kernels/data/window_dataset.cc", "CheckExternalState");
 return Status::OK(); }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_datasetDTcc mht_9(mht_9_v, 282, "", "./tensorflow/core/kernels/data/window_dataset.cc", "AsGraphDefInternal");

    if (ctx->is_graph_rewrite()) {
      // If data tensors are not to be serialized (e.g. when the serialization
      // is done for the sake of graph optimizations), we return
      // `errors::Unimplemented` to short-circuit the computation.
      return errors::Unimplemented(DebugString(),
                                   " does not support serialization");
    }
    std::vector<Node*> input_nodes;
    for (const auto& element : elements_) {
      for (const auto& t : element) {
        Node* node;
        TF_RETURN_IF_ERROR(b->AddDatasetOrTensor(ctx, t, &node));
        input_nodes.emplace_back(node);
      }
    }
    TF_RETURN_IF_ERROR(
        b->AddDataset(this, {}, {std::make_pair(0, input_nodes)}, {}, output));
    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Window> {
   public:
    explicit Iterator(const Params& params) : DatasetIterator<Window>(params) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_datasetDTcc mht_10(mht_10_v, 309, "", "./tensorflow/core/kernels/data/window_dataset.cc", "Iterator");
}

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_datasetDTcc mht_11(mht_11_v, 316, "", "./tensorflow/core/kernels/data/window_dataset.cc", "GetNextInternal");

      mutex_lock l(mu_);
      if (i_ == dataset()->elements_.size()) {
        *end_of_sequence = true;
      } else {
        *end_of_sequence = false;
        *out_tensors = dataset()->elements_[i_++];
      }
      return Status::OK();
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_datasetDTcc mht_12(mht_12_v, 331, "", "./tensorflow/core/kernels/data/window_dataset.cc", "SaveInternal");

      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kCurIndex), i_));
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_datasetDTcc mht_13(mht_13_v, 341, "", "./tensorflow/core/kernels/data/window_dataset.cc", "RestoreInternal");

      mutex_lock l(mu_);
      int64_t i;
      TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kCurIndex), &i));
      i_ = size_t(i);
      return Status::OK();
    }

    mutex mu_;
    size_t i_ TF_GUARDED_BY(mu_) = 0;
  };

  const std::vector<std::vector<Tensor>> elements_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
};

class WindowOp : public DatasetOpKernel {
 public:
  explicit WindowOp(OpKernelConstruction* ctx) : DatasetOpKernel(ctx) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_datasetDTcc mht_14(mht_14_v, 363, "", "./tensorflow/core/kernels/data/window_dataset.cc", "WindowOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
  }

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_datasetDTcc mht_15(mht_15_v, 372, "", "./tensorflow/core/kernels/data/window_dataset.cc", "MakeDataset");

    OpInputList inputs;
    OP_REQUIRES_OK(ctx, ctx->input_list(kInputs, &inputs));
    auto element_size = output_shapes_.size();
    auto num_elements = ctx->num_inputs() / element_size;
    std::vector<std::vector<Tensor>> elements;
    for (size_t i = 0; i < num_elements; ++i) {
      std::vector<Tensor> element;
      for (size_t j = 0; j < element_size; ++j) {
        element.push_back(std::move(inputs[i * element_size + j]));
      }
      elements.push_back(std::move(element));
    }
    *output = new Window(std::move(elements), output_types_, output_shapes_);
  }

 private:
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

REGISTER_KERNEL_BUILDER(Name("WindowOp").Device(DEVICE_CPU), WindowOp);

}  // namespace

Status NewWindow(std::vector<std::vector<Tensor>> elements,
                 DataTypeVector output_types,
                 std::vector<PartialTensorShape> output_shapes,
                 DatasetBase** out_dataset) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_datasetDTcc mht_16(mht_16_v, 403, "", "./tensorflow/core/kernels/data/window_dataset.cc", "NewWindow");

  // TODO(mrry): If this becomes more public, we must validate that
  // the elements match the output_types and output_shapes.
  *out_dataset = new Window(std::move(elements), std::move(output_types),
                            std::move(output_shapes));
  (*out_dataset)->Initialize(/*metadata=*/{});
  return Status::OK();
}

}  // namespace data
}  // namespace tensorflow
