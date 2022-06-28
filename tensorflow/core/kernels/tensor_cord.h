/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_TENSOR_CORD_H_
#define TENSORFLOW_CORE_KERNELS_TENSOR_CORD_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPStensor_cordDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_cordDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPStensor_cordDTh() {
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


#include <array>
#include <numeric>

#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/core/framework/variant_tensor_data.h"

namespace tensorflow {

typedef void (*CordRepReleaser)(void*);

class TensorCord {
  // A TensorCord keeps a view into some data, and a cleanup method to clean up
  // that data when the TensorCord destructor is called.  Copying a TensorCord
  // increments a reference count to the cleanup method, and so the cleanup
  // method is only called when all copies of the original TensorCord are
  // cleared.
  //
  // Example:
  //
  // const string& s = t.scalar<string>()();
  // TensorCord tc(s, &t);
  // ASSERT_EQ(s, tc.view());
  // TensorCord copy(tc);
  // tc = TensorCord();  // cleanup not called; the reference is held by `copy`.
  // copy = TensorCord();  // cleanup happens now, the reference is destroyed.
  //
  // Another example:
  //
  // void TensorProtoDeleter(void* ptr) {
  //   delete static_cast<TensorProto*>(ptr);
  // }
  //
  // auto p = absl::MakeUnique<TensorProto>(...);
  // absl::string_view content(p->tensor_content());
  // TensorCord tc(content, TensorProtoDeleter, p.release());
  //

 public:
  static constexpr const char kTypeName[] = "tensorflow::TensorCord";

  TensorCord() : chunks_() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_cordDTh mht_0(mht_0_v, 231, "", "./tensorflow/core/kernels/tensor_cord.h", "TensorCord");
}

  ~TensorCord();

  // Args:
  //   `view`: should point to a location in memory that is guaranteed to remain
  //           valid until `releaser` is called.
  //   `releaser`: A callback that will be executed when there are no references
  //               left on `view`.  It will be called via `releaser(memory)`.
  //   `memory`: The argument passed to `releaser` when it is called.
  //
  // You are STRONGLY advised to provide a non-null `releaser`, and a pointer
  // to the underlying data (while ensuring that the data will not be deleted
  // until `releaser(memory)` is called).  Otherwise the TensorCord may
  // outlive the data backing `view`.
  TensorCord(absl::string_view view, CordRepReleaser releaser,
             void* memory = nullptr)
      : chunks_({new CordRep(view, releaser, memory)}) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("view: \"" + std::string(view.data(), view.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_cordDTh mht_1(mht_1_v, 252, "", "./tensorflow/core/kernels/tensor_cord.h", "TensorCord");
}

  // Args:
  //   `view`: should point to a location in memory backed by `tensor`,
  //      e.g., `view` is a string_view on a tstring which is an element
  //      of `tensor`.  Furthermore, the associated tstring is not expected
  //      to be modified in such a way that the underlying memory will
  //      be changed after this TensorCord is created.
  TensorCord(absl::string_view view, Tensor* tensor)
      : chunks_({NewCordRepFromTensor(view, tensor)}) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("view: \"" + std::string(view.data(), view.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_cordDTh mht_2(mht_2_v, 265, "", "./tensorflow/core/kernels/tensor_cord.h", "TensorCord");
}

  // Disallow construction with empty callback or empty tensor.
  TensorCord(absl::string_view view, std::nullptr_t, void* memory) = delete;
  TensorCord(absl::string_view view, std::nullptr_t) = delete;

  TensorCord(const TensorCord& other);

  TensorCord(TensorCord&& other) noexcept;

  TensorCord& operator=(const TensorCord& other);

  TensorCord& operator=(TensorCord&& other) noexcept;

  void Append(const TensorCord& other);

  void Append(absl::string_view view, CordRepReleaser releaser,
              void* memory = nullptr);

  void Append(absl::string_view view, Tensor* tensor);

  // Disallow Appends with empty callbacks or empty tensors.
  void Append(absl::string_view view, std::nullptr_t, void* memory) = delete;
  void Append(absl::string_view view, std::nullptr_t) = delete;

  size_t size() const;
  bool empty() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_cordDTh mht_3(mht_3_v, 294, "", "./tensorflow/core/kernels/tensor_cord.h", "empty");
 return size() == 0; }

  // NOTE: This performs an expensive copy of the underlying data.
  explicit operator string() const;

  class ChunkIterator {
   public:
    using iterator_category = std::input_iterator_tag;
    using value_type = absl::string_view;
    using difference_type = ptrdiff_t;
    using pointer = const value_type*;
    using reference = value_type;

    ChunkIterator& operator++();

    ChunkIterator operator++(int) {
      ChunkIterator tmp(*this);
      operator++();
      return tmp;
    }

    bool operator==(const ChunkIterator& other) const {
      return (cord_ == other.cord_ && chunk_index_ == other.chunk_index_);
    }

    bool operator!=(const ChunkIterator& other) const {
      return !(*this == other);
    }
    reference operator*() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_cordDTh mht_4(mht_4_v, 325, "", "./tensorflow/core/kernels/tensor_cord.h", "*");

      assert(cord_ != nullptr);
      return view_;
    }
    pointer operator->() const {
      assert(cord_ != nullptr);
      return &view_;
    }

    friend class TensorCord;

   private:
    // Constructs a `begin()` iterator from `cord`.
    explicit ChunkIterator(const TensorCord* cord, int chunk_index);

    const TensorCord* const cord_;
    int chunk_index_;
    absl::string_view view_;
  };

  class ChunkRange {
   public:
    explicit ChunkRange(const TensorCord* cord) : cord_(cord) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_cordDTh mht_5(mht_5_v, 350, "", "./tensorflow/core/kernels/tensor_cord.h", "ChunkRange");
}

    ChunkIterator begin() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_cordDTh mht_6(mht_6_v, 355, "", "./tensorflow/core/kernels/tensor_cord.h", "begin");
 return ChunkIterator(cord_, 0); }

    ChunkIterator end() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_cordDTh mht_7(mht_7_v, 360, "", "./tensorflow/core/kernels/tensor_cord.h", "end");

      return ChunkIterator(cord_, cord_->chunks_.size());
    }

   private:
    const TensorCord* cord_;
  };

  // Note that the ordinary caveats of temporary lifetime extension apply:
  //
  //   void Process() {
  //     for (absl::string_view chunk : CordFactory().Chunks()) {
  //       // The temporary Cord returned by CordFactory has been destroyed!
  //     }
  //   }
  ChunkRange Chunks() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_cordDTh mht_8(mht_8_v, 378, "", "./tensorflow/core/kernels/tensor_cord.h", "Chunks");
 return ChunkRange(this); }

  ChunkIterator chunk_begin() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_cordDTh mht_9(mht_9_v, 383, "", "./tensorflow/core/kernels/tensor_cord.h", "chunk_begin");
 return ChunkIterator(this, 0); }

  ChunkIterator chunk_end() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_cordDTh mht_10(mht_10_v, 388, "", "./tensorflow/core/kernels/tensor_cord.h", "chunk_end");

    return ChunkIterator(this, chunks_.size());
  }

  static string TypeName() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_cordDTh mht_11(mht_11_v, 395, "", "./tensorflow/core/kernels/tensor_cord.h", "TypeName");
 return kTypeName; }

  string DebugString() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_cordDTh mht_12(mht_12_v, 400, "", "./tensorflow/core/kernels/tensor_cord.h", "DebugString");

    return absl::StrCat("<TensorCord size=", size(), ">");
  }

  void Encode(VariantTensorData* data) const;

  bool Decode(VariantTensorData data);

 private:
  void Cleanup();

  class CordRep : public core::RefCounted {
   public:
    CordRep(absl::string_view view, CordRepReleaser releaser,
            void* arg = nullptr)
        : is_inline_(false), rep_(view, releaser, arg) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("view: \"" + std::string(view.data(), view.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_cordDTh mht_13(mht_13_v, 419, "", "./tensorflow/core/kernels/tensor_cord.h", "CordRep");
}

    // **WARNING** Only use this constructor if
    //    view.size() < CordRep::kMaxInlineSize.
    explicit CordRep(absl::string_view view) : is_inline_(true), rep_(view) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("view: \"" + std::string(view.data(), view.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_cordDTh mht_14(mht_14_v, 427, "", "./tensorflow/core/kernels/tensor_cord.h", "CordRep");
}

    ~CordRep() override;

    absl::string_view view() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_cordDTh mht_15(mht_15_v, 434, "", "./tensorflow/core/kernels/tensor_cord.h", "view");

      if (is_inline_) {
        return absl::string_view(
            rep_.internal.data() + 1,
            *reinterpret_cast<const uint8*>(rep_.internal.data()));
      } else {
        return rep_.external.view;
      }
    }

   private:
    friend class TensorCord;

    struct ExternalRep {
      absl::string_view view;
      CordRepReleaser releaser;
      void* arg;

      ExternalRep(absl::string_view view_, CordRepReleaser releaser_,
                  void* arg_)
          : view(view_), releaser(releaser_), arg(arg_) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("view_: \"" + std::string(view_.data(), view_.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_cordDTh mht_16(mht_16_v, 458, "", "./tensorflow/core/kernels/tensor_cord.h", "ExternalRep");
}
    };

    // We save the size in the first byte, so subtract 1.
    static constexpr int kMaxInlineSize = sizeof(ExternalRep) - 1;
    static_assert(kMaxInlineSize < 255,
                  "Cannot store size of InlineRep in a single byte.");

    // The first byte stores the size as a uint8.  The rest of the bytes are the
    // string itself.
    using InlineRep = std::array<char, sizeof(ExternalRep)>;

    // Member variables.
    const bool is_inline_;
    const union _rep_union {
      InlineRep internal;
      ExternalRep external;

      _rep_union(absl::string_view view, CordRepReleaser releaser, void* arg)
          : external(view, releaser, arg) {}

      explicit _rep_union(absl::string_view view) {
        DCHECK_LT(view.size(), kMaxInlineSize);
        *reinterpret_cast<uint8*>(internal.data()) = view.size();
        std::memcpy(static_cast<char*>(internal.data() + 1), view.data(),
                    view.size());
      }
    } rep_;
  };

  static TensorBuffer* TensorBufWithRef(Tensor* tensor);
  static void TensorBufReleaser(void* tensor_buffer);
  static void StringReleaser(void* str_ptr);
  static CordRep* NewCordRepFromTensor(absl::string_view view, Tensor* tensor);

  absl::InlinedVector<CordRep*, 2> chunks_;
};

inline TensorCord::TensorCord(const TensorCord& other)
    : chunks_(other.chunks_) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_cordDTh mht_17(mht_17_v, 500, "", "./tensorflow/core/kernels/tensor_cord.h", "TensorCord::TensorCord");

  for (auto* rep : chunks_) {
    rep->Ref();
  }
}

inline TensorCord::TensorCord(TensorCord&& other) noexcept
    : chunks_(std::move(other.chunks_)) {
  other.chunks_.clear();
}

inline TensorCord& TensorCord::operator=(const TensorCord& other) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_cordDTh mht_18(mht_18_v, 514, "", "./tensorflow/core/kernels/tensor_cord.h", "=");

  Cleanup();
  chunks_ = other.chunks_;
  for (auto* rep : chunks_) {
    rep->Ref();
  }
  return *this;
}

inline TensorCord& TensorCord::operator=(TensorCord&& other) noexcept {
  Cleanup();
  std::swap(chunks_, other.chunks_);
  return *this;
}

inline void TensorCord::Append(const TensorCord& other) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_cordDTh mht_19(mht_19_v, 532, "", "./tensorflow/core/kernels/tensor_cord.h", "TensorCord::Append");

  for (auto* rep : other.chunks_) {
    chunks_.push_back(rep);
    rep->Ref();
  }
}

inline void TensorCord::Append(absl::string_view view, CordRepReleaser releaser,
                               void* memory) {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("view: \"" + std::string(view.data(), view.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_cordDTh mht_20(mht_20_v, 544, "", "./tensorflow/core/kernels/tensor_cord.h", "TensorCord::Append");

  chunks_.push_back(new CordRep(view, releaser, memory));
}

inline void TensorCord::Append(absl::string_view view, Tensor* tensor) {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("view: \"" + std::string(view.data(), view.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_cordDTh mht_21(mht_21_v, 552, "", "./tensorflow/core/kernels/tensor_cord.h", "TensorCord::Append");

  chunks_.push_back(NewCordRepFromTensor(view, tensor));
}

inline size_t TensorCord::size() const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_cordDTh mht_22(mht_22_v, 559, "", "./tensorflow/core/kernels/tensor_cord.h", "TensorCord::size");

  return (chunks_.empty())
             ? 0
             : std::accumulate(chunk_begin(), chunk_end(), 0,
                               [](size_t acc, absl::string_view b) {
                                 return acc + b.size();
                               });
}

inline TensorCord::ChunkIterator& TensorCord::ChunkIterator::operator++() {
  assert(cord_ != nullptr);
  assert(chunk_index_ < cord_->chunks_.size());
  chunk_index_ += 1;
  if (chunk_index_ != cord_->chunks_.size()) {
    view_ = cord_->chunks_[chunk_index_]->view();
  }
  return *this;
}

inline TensorCord::ChunkIterator::ChunkIterator(const TensorCord* cord,
                                                int index)
    : cord_(cord), chunk_index_(index) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_cordDTh mht_23(mht_23_v, 583, "", "./tensorflow/core/kernels/tensor_cord.h", "TensorCord::ChunkIterator::ChunkIterator");

  if (index < cord_->chunks_.size()) {
    view_ = cord_->chunks_[index]->view();
  }
}

inline TensorCord::CordRep* TensorCord::NewCordRepFromTensor(
    absl::string_view view, Tensor* tensor) {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("view: \"" + std::string(view.data(), view.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_cordDTh mht_24(mht_24_v, 594, "", "./tensorflow/core/kernels/tensor_cord.h", "TensorCord::NewCordRepFromTensor");

  if (view.size() <= TensorCord::CordRep::kMaxInlineSize) {
    return new CordRep(view);
  } else {
    return new CordRep(view, &TensorBufReleaser, TensorBufWithRef(tensor));
  }
}

inline void TensorCord::Cleanup() {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_cordDTh mht_25(mht_25_v, 605, "", "./tensorflow/core/kernels/tensor_cord.h", "TensorCord::Cleanup");

  if (chunks_.empty()) return;
  for (auto* rep : chunks_) {
    rep->Unref();
  }
  chunks_.clear();
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_TENSOR_CORD_H_
