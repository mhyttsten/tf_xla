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
#ifndef TENSORFLOW_CORE_KERNELS_TENSOR_MAP_H_
#define TENSORFLOW_CORE_KERNELS_TENSOR_MAP_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPStensor_mapDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_mapDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPStensor_mapDTh() {
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

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_key.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/lib/core/refcount.h"

namespace tensorflow {

// Variant compatible type for a map of tensors. This is mutable but instances
// should never be mutated after stored in a variant tensor.
//
// **NOTE**: TensorMap stores a refcounted container of tf::Tensor objects,
// which are accessible via TensorMap::tensors().  Because it is refcounted,
// straight copies of the form:
//
//    TensorMap b = a;
//    b.tensors().insert(k,v);  // WARNING: This modifies a.tensors().
//
// Do not create a true copy of the underlying container - but instead increment
// a reference count.  Modifying b.tensors() modifies a.tensors().  In this way,
// TensorMap should be considered similar to the tf::Tensor object.
//
// In order to get a copy of the underlying map, use the Copy method:
//
//    TensorMap b = a.Copy();
//    b.tensors().insert(k, v);  // This does not modify a.tensors().
//
// Note that this is not a deep copy: the memory locations of the underlying
// tensors will still point to the same locations of the corresponding tensors
// in the original.  To truly perform a deep copy, Device and Type-specific
// code needs to be applied to the underlying tensors as usual.
//
// The most important implication of RefCounted TensorMaps is that OpKernels
// wishing to reuse TensorMap inputs as outputs via context->forward_input()
// need to perform an additional check on the refcount of the TensorList,
// to ensure aliasing can be performed safely.  For example:
//
//     bool can_alias = false;
//     auto fw = c->forward_input(..., DT_VARIANT, {}, ...);
//     if (fw && fw->dtype() == DT_VARIANT && fw->NumElements() == 1) {
//       auto* tl = fw->scalar<Variant>()().get<TensorMap>();
//       if (tl && tl->RefCountIsOne()) {
//         can_alias = true;
//       }
//     }
//
class TensorMap {
 public:
  TensorMap() : tensors_(new Tensors) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_mapDTh mht_0(mht_0_v, 238, "", "./tensorflow/core/kernels/tensor_map.h", "TensorMap");
}
  ~TensorMap();

  TensorMap(const TensorMap& other) : tensors_(other.tensors_) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_mapDTh mht_1(mht_1_v, 244, "", "./tensorflow/core/kernels/tensor_map.h", "TensorMap");

    tensors_->Ref();
  }

  TensorMap(TensorMap&& rhs) : tensors_(rhs.tensors_) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_mapDTh mht_2(mht_2_v, 251, "", "./tensorflow/core/kernels/tensor_map.h", "TensorMap");

    rhs.tensors_ = nullptr;
  }

  TensorMap& operator=(const TensorMap& rhs) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_mapDTh mht_3(mht_3_v, 258, "", "./tensorflow/core/kernels/tensor_map.h", "=");

    if (this == &rhs) return *this;
    tensors_->Unref();
    tensors_ = rhs.tensors_;
    tensors_->Ref();
    return *this;
  }

  TensorMap& operator=(TensorMap&& rhs) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_mapDTh mht_4(mht_4_v, 269, "", "./tensorflow/core/kernels/tensor_map.h", "=");

    if (this == &rhs) return *this;
    std::swap(tensors_, rhs.tensors_);
    return *this;
  }

  static const char kTypeName[];

  string TypeName() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_mapDTh mht_5(mht_5_v, 280, "", "./tensorflow/core/kernels/tensor_map.h", "TypeName");
 return kTypeName; }

  void Encode(VariantTensorData* data) const;

  bool Decode(const VariantTensorData& data);

  // TODO(apassos) fill this out
  string DebugString() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_mapDTh mht_6(mht_6_v, 290, "", "./tensorflow/core/kernels/tensor_map.h", "DebugString");
 return "TensorMap"; }

  // Access to the underlying tensor container.
  absl::flat_hash_map<TensorKey, Tensor>& tensors() {
    return tensors_->values_;
  }

  const absl::flat_hash_map<TensorKey, Tensor>& tensors() const {
    return tensors_->values_;
  }

  // Get a new TensorMap containing a copy of the underlying tensor container.
  TensorMap Copy() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_mapDTh mht_7(mht_7_v, 305, "", "./tensorflow/core/kernels/tensor_map.h", "Copy");

    TensorMap out;
    // This performs a copy of the absl::hashmap.
    out.tensors_->values_ = tensors_->values_;
    return out;
  }

  // Insert key and value if the key does not already exist.
  // Returns true if the insertion happens.
  bool insert(const TensorKey& key, const Tensor& value) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_mapDTh mht_8(mht_8_v, 317, "", "./tensorflow/core/kernels/tensor_map.h", "insert");

    auto r = tensors_->values_.try_emplace(key, value);
    return r.second;
  }

  // Lookup given key. Returns iterator to found key or end.
  absl::flat_hash_map<TensorKey, Tensor>::iterator find(TensorKey key) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_mapDTh mht_9(mht_9_v, 326, "", "./tensorflow/core/kernels/tensor_map.h", "find");

    return tensors_->values_.find(key);
  }

  Tensor& lookup(TensorKey key) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_mapDTh mht_10(mht_10_v, 333, "", "./tensorflow/core/kernels/tensor_map.h", "lookup");
 return tensors_->values_.find(key)->second; }

  Tensor& operator[](TensorKey& k) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_mapDTh mht_11(mht_11_v, 338, "", "./tensorflow/core/kernels/tensor_map.h", "lambda");
 return tensors_->values_[k]; }

  bool replace(const TensorKey& k, const Tensor& v) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_mapDTh mht_12(mht_12_v, 343, "", "./tensorflow/core/kernels/tensor_map.h", "replace");

    tensors_->values_[k] = v;
    return true;
  }

  // Removes element with given key. Return size of removed element.
  size_t erase(TensorKey key) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_mapDTh mht_13(mht_13_v, 352, "", "./tensorflow/core/kernels/tensor_map.h", "erase");
 return tensors_->values_.erase(key); }

  // Size returns the number of elements in the map
  size_t size() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_mapDTh mht_14(mht_14_v, 358, "", "./tensorflow/core/kernels/tensor_map.h", "size");
 return tensors_->values_.size(); }

  std::vector<Tensor> keys() const {
    std::vector<Tensor> keys;
    keys.reserve(tensors_->values_.size());
    absl::flat_hash_map<TensorKey, Tensor>::iterator it =
        tensors_->values_.begin();
    while (it != tensors_->values_.end()) {
      keys.push_back(it->first);
      it++;
    }
    return keys;
  }

  // Is this TensorMap the only one with a reference to the underlying
  // container?
  bool RefCountIsOne() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_mapDTh mht_15(mht_15_v, 377, "", "./tensorflow/core/kernels/tensor_map.h", "RefCountIsOne");
 return tensors_->RefCountIsOne(); }

 private:
  class Tensors : public core::RefCounted {
   public:
    absl::flat_hash_map<TensorKey, Tensor> values_;
  };
  Tensors* tensors_;
};

#if defined(PLATFORM_GOOGLE)
// TODO(ebrevdo): Identify why Variant inline size is smaller on mobile devices.
// For 32-bit devices, it's acceptable not to inline.
static_assert(Variant::CanInlineType<TensorMap>() || sizeof(void*) < 8,
              "Must be able to inline TensorMap into a Variant");
#endif
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_TENSOR_MAP_H_
