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
#ifndef TENSORFLOW_CORE_KERNELS_TENSOR_LIST_H_
#define TENSORFLOW_CORE_KERNELS_TENSOR_LIST_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPStensor_listDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_listDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPStensor_listDTh() {
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

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/lib/core/refcount.h"

namespace tensorflow {

// Variant compatible type for a list of tensors. This is mutable but instances
// should never be mutated after stored in a variant tensor.
//
// **NOTE**: TensorList stores a refcounted container of tf::Tensor objects,
// which are accessible via TensorList::tensors().  Because it is refcounted,
// straight copies of the form:
//
//    TensorList b = a;
//    b.tensors().push_back(t);  // WARNING: This modifies a.tensors().
//
// Do not create a true copy of the underlying container - but instead increment
// a reference count.  Modifying b.tensors() modifies a.tensors().  In this way,
// TensorList should be considered similar to the tf::Tensor object.
//
// In order to get a copy of the underlying list, use the Copy method:
//
//    TensorList b = a.Copy();
//    b.tensors().push_back(t);  // This does not modify a.tensors().
//
// Note that this is not a deep copy: the memory locations of the underlying
// tensors will still point to the same locations of the corresponding tensors
// in the original.  To truly perform a deep copy, Device and Type-specific
// code needs to be applied to the underlying tensors as usual.
//
// The most important implication of RefCounted TLs is that OpKernels
// wishing to reuse TensorList inputs as outputs via context->forward_input()
// need to perform an additional check on the refcount of the TensorList,
// to ensure aliasing can be performed safely.  For example:
//
//     bool can_alias = false;
//     auto fw = c->forward_input(..., DT_VARIANT, {}, ...);
//     if (fw && fw->dtype() == DT_VARIANT && fw->NumElements() == 1) {
//       auto* tl = fw->scalar<Variant>()().get<TensorList>();
//       if (tl && tl->RefCountIsOne()) {
//         can_alias = true;
//       }
//     }
//
class TensorList {
 public:
  TensorList() : tensors_(new Tensors) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_listDTh mht_0(mht_0_v, 236, "", "./tensorflow/core/kernels/tensor_list.h", "TensorList");
}
  ~TensorList();

  TensorList(const TensorList& other)
      : element_shape(other.element_shape),
        element_dtype(other.element_dtype),
        max_num_elements(other.max_num_elements),
        tensors_(other.tensors_) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_listDTh mht_1(mht_1_v, 246, "", "./tensorflow/core/kernels/tensor_list.h", "TensorList");

    tensors_->Ref();
  }

  TensorList(TensorList&& rhs)
      : element_shape(std::move(rhs.element_shape)),
        element_dtype(rhs.element_dtype),
        max_num_elements(rhs.max_num_elements),
        tensors_(rhs.tensors_) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_listDTh mht_2(mht_2_v, 257, "", "./tensorflow/core/kernels/tensor_list.h", "TensorList");

    rhs.tensors_ = nullptr;
  }

  TensorList& operator=(const TensorList& rhs) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_listDTh mht_3(mht_3_v, 264, "", "./tensorflow/core/kernels/tensor_list.h", "=");

    if (this == &rhs) return *this;
    element_shape = rhs.element_shape;
    element_dtype = rhs.element_dtype;
    max_num_elements = rhs.max_num_elements;
    tensors_->Unref();
    tensors_ = rhs.tensors_;
    tensors_->Ref();
    return *this;
  }

  TensorList& operator=(TensorList&& rhs) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_listDTh mht_4(mht_4_v, 278, "", "./tensorflow/core/kernels/tensor_list.h", "=");

    if (this == &rhs) return *this;
    element_shape = rhs.element_shape;
    element_dtype = rhs.element_dtype;
    max_num_elements = rhs.max_num_elements;
    std::swap(tensors_, rhs.tensors_);
    return *this;
  }

  static const char kTypeName[];

  string TypeName() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_listDTh mht_5(mht_5_v, 292, "", "./tensorflow/core/kernels/tensor_list.h", "TypeName");
 return kTypeName; }

  void Encode(VariantTensorData* data) const;

  bool Decode(const VariantTensorData& data);

  // TODO(apassos) fill this out
  string DebugString() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_listDTh mht_6(mht_6_v, 302, "", "./tensorflow/core/kernels/tensor_list.h", "DebugString");
 return "TensorList"; }

  PartialTensorShape element_shape;

  DataType element_dtype;

  // The maximum allowed size of `tensors`. Defaults to -1 meaning that the size
  // of `tensors` is unbounded.
  int max_num_elements = -1;

  // Access to the underlying tensor container.
  std::vector<Tensor>& tensors() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_listDTh mht_7(mht_7_v, 316, "", "./tensorflow/core/kernels/tensor_list.h", "tensors");
 return tensors_->values_; }
  const std::vector<Tensor>& tensors() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_listDTh mht_8(mht_8_v, 320, "", "./tensorflow/core/kernels/tensor_list.h", "tensors");
 return tensors_->values_; }

  // Get a new TensorList containing a copy of the underlying tensor container.
  TensorList Copy() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_listDTh mht_9(mht_9_v, 326, "", "./tensorflow/core/kernels/tensor_list.h", "Copy");

    TensorList out;
    out.element_shape = element_shape;
    out.element_dtype = element_dtype;
    out.max_num_elements = max_num_elements;
    // This performs a copy of the std::vector.
    out.tensors_->values_ = tensors_->values_;
    return out;
  }

  // Is this TensorList the only one with a reference to the underlying
  // container?
  bool RefCountIsOne() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_listDTh mht_10(mht_10_v, 341, "", "./tensorflow/core/kernels/tensor_list.h", "RefCountIsOne");
 return tensors_->RefCountIsOne(); }

 private:
  class Tensors : public core::RefCounted {
   public:
    std::vector<Tensor> values_;
  };
  Tensors* tensors_;
};

#if defined(PLATFORM_GOOGLE)
// TODO(ebrevdo): Identify why Variant inline size is smaller on mobile devices.
// For 32-bit devices, it's acceptable not to inline.
static_assert(Variant::CanInlineType<TensorList>() || sizeof(void*) < 8,
              "Must be able to inline TensorList into a Variant");
#endif
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_TENSOR_LIST_H_
