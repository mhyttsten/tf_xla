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
class MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc() {
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

#include "tensorflow/core/framework/variant.h"

#include <cstddef>
#if defined(__x86_64__)
#include <xmmintrin.h>
#endif
#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

namespace {

template <typename T, bool BIG>
struct Wrapper {
  T value;
  char big[BIG ? 256 : 1];
  string TypeName() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/framework/variant_test.cc", "TypeName");
 return "POD"; }
};

template <bool BIG>
using Int = Wrapper<int, BIG>;

template <bool BIG>
using Float = Wrapper<float, BIG>;

template <bool BIG>
class MaybeAlive {
 public:
  MaybeAlive() : alive_(false) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_1(mht_1_v, 225, "", "./tensorflow/core/framework/variant_test.cc", "MaybeAlive");
}

  explicit MaybeAlive(bool alive) : alive_(alive) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_2(mht_2_v, 230, "", "./tensorflow/core/framework/variant_test.cc", "MaybeAlive");

    if (alive) ++live_counter_;
  }

  ~MaybeAlive() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_3(mht_3_v, 237, "", "./tensorflow/core/framework/variant_test.cc", "~MaybeAlive");

    if (alive_) --live_counter_;
  }

  MaybeAlive(const MaybeAlive& rhs) : alive_(rhs.alive_) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_4(mht_4_v, 244, "", "./tensorflow/core/framework/variant_test.cc", "MaybeAlive");

    if (alive_) ++live_counter_;
  }

  MaybeAlive& operator=(const MaybeAlive& rhs) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_5(mht_5_v, 251, "", "./tensorflow/core/framework/variant_test.cc", "=");

    if (this == &rhs) return *this;
    if (alive_) --live_counter_;
    alive_ = rhs.alive_;
    if (alive_) ++live_counter_;
    return *this;
  }

  MaybeAlive(MaybeAlive&& rhs) : alive_(false) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_6(mht_6_v, 262, "", "./tensorflow/core/framework/variant_test.cc", "MaybeAlive");

    alive_ = std::move(rhs.alive_);
    if (alive_) ++live_counter_;
  }

  MaybeAlive& operator=(MaybeAlive&& rhs) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_7(mht_7_v, 270, "", "./tensorflow/core/framework/variant_test.cc", "=");

    if (this == &rhs) return *this;
    if (alive_) --live_counter_;
    alive_ = std::move(rhs.alive_);
    if (alive_) ++live_counter_;
    return *this;
  }

  static int LiveCounter() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_8(mht_8_v, 281, "", "./tensorflow/core/framework/variant_test.cc", "LiveCounter");
 return live_counter_; }

  string TypeName() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_9(mht_9_v, 286, "", "./tensorflow/core/framework/variant_test.cc", "TypeName");
 return "MaybeAlive"; }
  void Encode(VariantTensorData* data) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_10(mht_10_v, 290, "", "./tensorflow/core/framework/variant_test.cc", "Encode");
}
  bool Decode(VariantTensorData data) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_11(mht_11_v, 294, "", "./tensorflow/core/framework/variant_test.cc", "Decode");
 return false; }

 private:
  bool alive_;
  char big_[BIG ? 256 : 1];
  static int live_counter_;
};

template <>
int MaybeAlive<false>::live_counter_ = 0;
template <>
int MaybeAlive<true>::live_counter_ = 0;

template <bool BIG>
class DeleteCounter {
 public:
  DeleteCounter() : big_{}, counter_(nullptr) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_12(mht_12_v, 313, "", "./tensorflow/core/framework/variant_test.cc", "DeleteCounter");
}
  explicit DeleteCounter(int* counter) : big_{}, counter_(counter) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_13(mht_13_v, 317, "", "./tensorflow/core/framework/variant_test.cc", "DeleteCounter");
}
  ~DeleteCounter() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_14(mht_14_v, 321, "", "./tensorflow/core/framework/variant_test.cc", "~DeleteCounter");

    if (counter_) ++*counter_;
  }
  // Need custom move operations because int* just gets copied on move, but we
  // need to clear counter_ on move.
  DeleteCounter& operator=(const DeleteCounter& rhs) = default;
  DeleteCounter& operator=(DeleteCounter&& rhs) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_15(mht_15_v, 330, "", "./tensorflow/core/framework/variant_test.cc", "=");

    if (this == &rhs) return *this;
    counter_ = rhs.counter_;
    rhs.counter_ = nullptr;
    return *this;
  }
  DeleteCounter(DeleteCounter&& rhs) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_16(mht_16_v, 339, "", "./tensorflow/core/framework/variant_test.cc", "DeleteCounter");

    counter_ = rhs.counter_;
    rhs.counter_ = nullptr;
  }
  DeleteCounter(const DeleteCounter& rhs) = default;
  char big_[BIG ? 256 : 1];
  int* counter_;

  string TypeName() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_17(mht_17_v, 350, "", "./tensorflow/core/framework/variant_test.cc", "TypeName");
 return "DeleteCounter"; }
  void Encode(VariantTensorData* data) const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_18(mht_18_v, 354, "", "./tensorflow/core/framework/variant_test.cc", "Encode");
}
  bool Decode(VariantTensorData data) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_19(mht_19_v, 358, "", "./tensorflow/core/framework/variant_test.cc", "Decode");
 return false; }
};

}  // end namespace

TEST(VariantTest, MoveAndCopyBetweenBigAndSmall) {
  Variant x;
  int deleted_big = 0;
  int deleted_small = 0;
  x = DeleteCounter</*BIG=*/true>(&deleted_big);
  EXPECT_EQ(deleted_big, 0);
  x = DeleteCounter</*BIG=*/false>(&deleted_small);
  EXPECT_EQ(deleted_big, 1);
  EXPECT_EQ(deleted_small, 0);
  x = DeleteCounter</*BIG=*/true>(&deleted_big);
  EXPECT_EQ(deleted_big, 1);
  EXPECT_EQ(deleted_small, 1);
  x.clear();
  EXPECT_EQ(deleted_big, 2);
  EXPECT_EQ(deleted_small, 1);
  DeleteCounter</*BIG=*/true> big(&deleted_big);
  DeleteCounter</*BIG=*/false> small(&deleted_small);
  EXPECT_EQ(deleted_big, 2);
  EXPECT_EQ(deleted_small, 1);
  x = big;
  EXPECT_EQ(deleted_big, 2);
  EXPECT_EQ(deleted_small, 1);
  x = small;
  EXPECT_EQ(deleted_big, 3);
  EXPECT_EQ(deleted_small, 1);
  x = std::move(big);
  EXPECT_EQ(deleted_big, 3);
  EXPECT_EQ(deleted_small, 2);
  x = std::move(small);
  EXPECT_EQ(deleted_big, 4);
  EXPECT_EQ(deleted_small, 2);
  x.clear();
  EXPECT_EQ(deleted_big, 4);
  EXPECT_EQ(deleted_small, 3);
}

TEST(VariantTest, MoveAndCopyBetweenBigAndSmallVariants) {
  int deleted_big = 0;
  int deleted_small = 0;
  {
    Variant x = DeleteCounter</*BIG=*/true>(&deleted_big);
    Variant y = DeleteCounter</*BIG=*/false>(&deleted_small);
    EXPECT_EQ(deleted_big, 0);
    EXPECT_EQ(deleted_small, 0);
    x = y;
    EXPECT_EQ(deleted_big, 1);
    EXPECT_EQ(deleted_small, 0);
    x = x;
    EXPECT_EQ(deleted_big, 1);
    EXPECT_EQ(deleted_small, 0);
    EXPECT_NE(x.get<DeleteCounter<false>>(), nullptr);
    EXPECT_NE(y.get<DeleteCounter<false>>(), nullptr);
    x = std::move(y);
    EXPECT_EQ(deleted_small, 1);
    EXPECT_NE(x.get<DeleteCounter<false>>(), nullptr);
  }
  EXPECT_EQ(deleted_big, 1);
  EXPECT_EQ(deleted_small, 2);

  deleted_big = 0;
  deleted_small = 0;
  {
    Variant x = DeleteCounter</*BIG=*/false>(&deleted_small);
    Variant y = DeleteCounter</*BIG=*/true>(&deleted_big);
    EXPECT_EQ(deleted_big, 0);
    EXPECT_EQ(deleted_small, 0);
    x = y;
    EXPECT_EQ(deleted_big, 0);
    EXPECT_EQ(deleted_small, 1);
    x = x;
    EXPECT_EQ(deleted_big, 0);
    EXPECT_EQ(deleted_small, 1);
    EXPECT_NE(x.get<DeleteCounter<true>>(), nullptr);
    EXPECT_NE(y.get<DeleteCounter<true>>(), nullptr);
    x = std::move(y);
    EXPECT_EQ(deleted_big, 1);
    EXPECT_NE(x.get<DeleteCounter<true>>(), nullptr);
  }
  EXPECT_EQ(deleted_big, 2);
  EXPECT_EQ(deleted_small, 1);
}

namespace {

template <bool BIG>
class MoveAndCopyCounter {
 public:
  MoveAndCopyCounter()
      : big_{}, move_counter_(nullptr), copy_counter_(nullptr) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_20(mht_20_v, 454, "", "./tensorflow/core/framework/variant_test.cc", "MoveAndCopyCounter");
}
  explicit MoveAndCopyCounter(int* move_counter, int* copy_counter)
      : big_{}, move_counter_(move_counter), copy_counter_(copy_counter) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_21(mht_21_v, 459, "", "./tensorflow/core/framework/variant_test.cc", "MoveAndCopyCounter");
}

  MoveAndCopyCounter& operator=(const MoveAndCopyCounter& rhs) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_22(mht_22_v, 464, "", "./tensorflow/core/framework/variant_test.cc", "=");

    copy_counter_ = rhs.copy_counter_;
    if (copy_counter_) ++*copy_counter_;
    return *this;
  }
  MoveAndCopyCounter& operator=(MoveAndCopyCounter&& rhs) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_23(mht_23_v, 472, "", "./tensorflow/core/framework/variant_test.cc", "=");

    move_counter_ = rhs.move_counter_;
    if (move_counter_) ++*move_counter_;
    return *this;
  }
  MoveAndCopyCounter(MoveAndCopyCounter&& rhs) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_24(mht_24_v, 480, "", "./tensorflow/core/framework/variant_test.cc", "MoveAndCopyCounter");

    move_counter_ = rhs.move_counter_;
    if (move_counter_) ++*move_counter_;
  }
  MoveAndCopyCounter(const MoveAndCopyCounter& rhs) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_25(mht_25_v, 487, "", "./tensorflow/core/framework/variant_test.cc", "MoveAndCopyCounter");

    copy_counter_ = rhs.copy_counter_;
    if (copy_counter_) ++*copy_counter_;
  }
  char big_[BIG ? 256 : 1];
  int* move_counter_;
  int* copy_counter_;

  string TypeName() const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_26(mht_26_v, 498, "", "./tensorflow/core/framework/variant_test.cc", "TypeName");
 return "MoveAndCopyCounter"; }
  void Encode(VariantTensorData* data) const {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_27(mht_27_v, 502, "", "./tensorflow/core/framework/variant_test.cc", "Encode");
}
  bool Decode(VariantTensorData data) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_28(mht_28_v, 506, "", "./tensorflow/core/framework/variant_test.cc", "Decode");
 return false; }
};

}  // namespace

TEST(VariantTest, EmplaceBigAndSmallVariants) {
  {
    int moved_big = 0;
    int moved_small = 0;
    int copied_big = 0;
    int copied_small = 0;
    Variant x = MoveAndCopyCounter</*BIG=*/true>(&moved_big, &copied_big);
    EXPECT_EQ(moved_big, 1);
    EXPECT_EQ(copied_big, 0);
    Variant y = MoveAndCopyCounter</*BIG=*/false>(&moved_small, &copied_small);
    EXPECT_EQ(moved_small, 1);
    EXPECT_EQ(copied_small, 0);
  }

  {
    int moved_big = 0;
    int moved_small = 0;
    int copied_big = 0;
    int copied_small = 0;
    Variant x(MoveAndCopyCounter</*BIG=*/true>(&moved_big, &copied_big));
    EXPECT_EQ(moved_big, 1);
    EXPECT_EQ(copied_big, 0);
    Variant y(MoveAndCopyCounter</*BIG=*/false>(&moved_small, &copied_small));
    EXPECT_EQ(moved_small, 1);
    EXPECT_EQ(copied_small, 0);
  }

  {
    int moved_big = 0;
    int moved_small = 0;
    int copied_big = 0;
    int copied_small = 0;
    Variant x;
    x.emplace<MoveAndCopyCounter</*BIG=*/true>>(&moved_big, &copied_big);
    EXPECT_EQ(moved_big, 0);
    EXPECT_EQ(copied_big, 0);
    Variant y;
    y.emplace<MoveAndCopyCounter</*BIG=*/false>>(&moved_small, &copied_small);
    EXPECT_EQ(moved_small, 0);
    EXPECT_EQ(copied_small, 0);
  }
}

template <bool BIG>
void TestDestructOnVariantMove() {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_29(mht_29_v, 558, "", "./tensorflow/core/framework/variant_test.cc", "TestDestructOnVariantMove");

  CHECK_EQ(MaybeAlive<BIG>::LiveCounter(), 0);
  {
    Variant a = MaybeAlive<BIG>(true);
    Variant b = std::move(a);
  }
  EXPECT_EQ(MaybeAlive<BIG>::LiveCounter(), 0);
}

TEST(VariantTest, RHSDestructOnVariantMoveBig) {
  TestDestructOnVariantMove</*BIG=*/true>();
}

TEST(VariantTest, RHSDestructOnVariantMoveSmall) {
  TestDestructOnVariantMove</*BIG=*/false>();
}

TEST(VariantTest, Int) {
  Variant x;
  EXPECT_EQ(x.get<void>(), nullptr);
  x = 3;
  EXPECT_NE(x.get<void>(), nullptr);
  EXPECT_EQ(*x.get<int>(), 3);
  EXPECT_EQ(x.TypeName(), "int");
}
#if defined(__x86_64__)
struct MayCreateAlignmentDifficulties {
  int a;
  __m128 b;
};

bool M128AllEqual(const __m128& a, const __m128& b) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_30(mht_30_v, 592, "", "./tensorflow/core/framework/variant_test.cc", "M128AllEqual");

  return _mm_movemask_ps(_mm_cmpeq_ps(a, b)) == 0xf;
}

TEST(VariantTest, NotAlignable) {
  Variant x;
  EXPECT_EQ(x.get<void>(), nullptr);
  __m128 v = _mm_set_ps(1.0, 2.0, 3.0, 4.0);
  x = MayCreateAlignmentDifficulties{-1, v};
  EXPECT_NE(x.get<void>(), nullptr);
  auto* x_val = x.get<MayCreateAlignmentDifficulties>();
  // check that *x_val == x
  Variant y = x;
  EXPECT_EQ(x_val->a, -1);
  EXPECT_TRUE(M128AllEqual(x_val->b, v));
  auto* y_val = y.get<MayCreateAlignmentDifficulties>();
  EXPECT_EQ(y_val->a, -1);
  EXPECT_TRUE(M128AllEqual(y_val->b, v));
  Variant z = std::move(y);
  auto* z_val = z.get<MayCreateAlignmentDifficulties>();
  EXPECT_EQ(z_val->a, -1);
  EXPECT_TRUE(M128AllEqual(z_val->b, v));
}
#endif
template <bool BIG>
void TestBasic() {
  Variant x;
  EXPECT_EQ(x.get<void>(), nullptr);

  x = Int<BIG>{42};

  EXPECT_NE(x.get<void>(), nullptr);
  EXPECT_NE(x.get<Int<BIG>>(), nullptr);
  EXPECT_EQ(x.get<Int<BIG>>()->value, 42);
  EXPECT_EQ(x.TypeName(), "POD");
}

TEST(VariantTest, Basic) { TestBasic<false>(); }

TEST(VariantTest, BasicBig) { TestBasic<true>(); }

template <bool BIG>
void TestConstGet() {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_31(mht_31_v, 637, "", "./tensorflow/core/framework/variant_test.cc", "TestConstGet");

  Variant x;
  EXPECT_EQ(x.get<void>(), nullptr);

  x = Int<BIG>{42};

  const Variant y = x;

  EXPECT_NE(y.get<void>(), nullptr);
  EXPECT_NE(y.get<Int<BIG>>(), nullptr);
  EXPECT_EQ(y.get<Int<BIG>>()->value, 42);
}

TEST(VariantTest, ConstGet) { TestConstGet<false>(); }

TEST(VariantTest, ConstGetBig) { TestConstGet<true>(); }

template <bool BIG>
void TestClear() {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_32(mht_32_v, 658, "", "./tensorflow/core/framework/variant_test.cc", "TestClear");

  Variant x;
  EXPECT_EQ(x.get<void>(), nullptr);

  x = Int<BIG>{42};

  EXPECT_NE(x.get<void>(), nullptr);
  EXPECT_NE(x.get<Int<BIG>>(), nullptr);
  EXPECT_EQ(x.get<Int<BIG>>()->value, 42);

  x.clear();
  EXPECT_EQ(x.get<void>(), nullptr);
}

TEST(VariantTest, Clear) { TestClear<false>(); }

TEST(VariantTest, ClearBig) { TestClear<true>(); }

template <bool BIG>
void TestClearDeletes() {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_33(mht_33_v, 680, "", "./tensorflow/core/framework/variant_test.cc", "TestClearDeletes");

  Variant x;
  EXPECT_EQ(x.get<void>(), nullptr);

  int deleted_count = 0;
  using DC = DeleteCounter<BIG>;
  DC dc(&deleted_count);
  EXPECT_EQ(deleted_count, 0);
  x = dc;
  EXPECT_EQ(deleted_count, 0);

  EXPECT_NE(x.get<void>(), nullptr);
  EXPECT_NE(x.get<DC>(), nullptr);

  x.clear();
  EXPECT_EQ(x.get<void>(), nullptr);
  EXPECT_EQ(deleted_count, 1);

  x = dc;
  EXPECT_EQ(deleted_count, 1);

  Variant y = x;
  EXPECT_EQ(deleted_count, 1);

  x.clear();
  EXPECT_EQ(deleted_count, 2);

  y.clear();
  EXPECT_EQ(deleted_count, 3);
}

TEST(VariantTest, ClearDeletesOnHeap) { TestClearDeletes</*BIG=*/true>(); }

TEST(VariantTest, ClearDeletesOnStack) { TestClearDeletes</*BIG=*/false>(); }

TEST(VariantTest, Tensor) {
  Variant x;
  Tensor t(DT_FLOAT, {});
  t.flat<float>()(0) = 42.0f;
  x = t;

  EXPECT_NE(x.get<Tensor>(), nullptr);
  EXPECT_EQ(x.get<Tensor>()->flat<float>()(0), 42.0f);
  x.get<Tensor>()->flat<float>()(0) += 1.0f;
  EXPECT_EQ(x.get<Tensor>()->flat<float>()(0), 43.0f);
  EXPECT_EQ(x.TypeName(), "tensorflow::Tensor");

  Tensor& foo_t = x.emplace<Tensor>("foo");
  EXPECT_NE(x.get<Tensor>(), nullptr);
  EXPECT_EQ(x.get<Tensor>()->scalar<tstring>()(), "foo");
  EXPECT_EQ(&foo_t, x.get<Tensor>());
  EXPECT_EQ(x.TypeName(), "tensorflow::Tensor");

  Tensor& bar_t = x.emplace<Tensor>(DT_INT64, TensorShape({1}));
  EXPECT_EQ(&bar_t, x.get<Tensor>());
  bar_t.vec<int64_t>()(0) = 17;
  EXPECT_EQ(x.get<Tensor>()->vec<int64_t>()(0), 17);
  bar_t.vec<int64_t>()(0) += 1;
  EXPECT_EQ(x.get<Tensor>()->vec<int64_t>()(0), 18);
}

TEST(VariantTest, NontrivialTensorVariantCopy) {
  Tensor variants(DT_VARIANT, {});
  Tensor t(true);
  test::FillValues<Variant>(&variants, gtl::ArraySlice<Variant>({t}));
  const Tensor* t_c = variants.flat<Variant>()(0).get<Tensor>();
  EXPECT_EQ(t_c->dtype(), t.dtype());
  EXPECT_EQ(t_c->shape(), t.shape());
  EXPECT_EQ(t_c->scalar<bool>()(), t.scalar<bool>()());
}

TEST(VariantTest, TensorProto) {
  Variant x;
  TensorProto t;
  t.set_dtype(DT_FLOAT);
  t.mutable_tensor_shape()->set_unknown_rank(true);
  x = t;

  EXPECT_EQ(x.TypeName(), "tensorflow.TensorProto");
  EXPECT_NE(x.get<TensorProto>(), nullptr);
  EXPECT_EQ(x.get<TensorProto>()->dtype(), DT_FLOAT);
  EXPECT_EQ(x.get<TensorProto>()->tensor_shape().unknown_rank(), true);
}

template <bool BIG>
void TestCopyValue() {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_34(mht_34_v, 768, "", "./tensorflow/core/framework/variant_test.cc", "TestCopyValue");

  Variant x, y;
  x = Int<BIG>{10};
  y = x;

  EXPECT_EQ(x.get<Int<BIG>>()->value, 10);
  EXPECT_EQ(x.get<Int<BIG>>()->value, y.get<Int<BIG>>()->value);
}

TEST(VariantTest, CopyValue) { TestCopyValue<false>(); }

TEST(VariantTest, CopyValueBig) { TestCopyValue<true>(); }

template <bool BIG>
void TestMoveValue() {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_35(mht_35_v, 785, "", "./tensorflow/core/framework/variant_test.cc", "TestMoveValue");

  Variant x;
  x = []() -> Variant {
    Variant y;
    y = Int<BIG>{10};
    return y;
  }();
  EXPECT_EQ(x.get<Int<BIG>>()->value, 10);
}

TEST(VariantTest, MoveValue) { TestMoveValue<false>(); }

TEST(VariantTest, MoveValueBig) { TestMoveValue<true>(); }

TEST(VariantTest, TypeMismatch) {
  Variant x;
  x = Int<false>{10};
  EXPECT_EQ(x.get<float>(), nullptr);
  EXPECT_EQ(x.get<int>(), nullptr);
  EXPECT_NE(x.get<Int<false>>(), nullptr);
}

struct TensorList {
  void Encode(VariantTensorData* data) const {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_36(mht_36_v, 811, "", "./tensorflow/core/framework/variant_test.cc", "Encode");
 data->tensors_ = vec; }

  bool Decode(VariantTensorData data) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_37(mht_37_v, 816, "", "./tensorflow/core/framework/variant_test.cc", "Decode");

    vec = std::move(data.tensors_);
    return true;
  }

  string TypeName() const {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_38(mht_38_v, 824, "", "./tensorflow/core/framework/variant_test.cc", "TypeName");
 return "TensorList"; }

  std::vector<Tensor> vec;
};

TEST(VariantTest, TensorListTest) {
  Variant x;

  TensorList vec;
  for (int i = 0; i < 4; ++i) {
    Tensor elem(DT_INT32, {1});
    elem.flat<int>()(0) = i;
    vec.vec.push_back(elem);
  }

  for (int i = 0; i < 4; ++i) {
    Tensor elem(DT_FLOAT, {1});
    elem.flat<float>()(0) = 2 * i;
    vec.vec.push_back(elem);
  }

  x = vec;

  EXPECT_EQ(x.TypeName(), "TensorList");
  EXPECT_EQ(x.DebugString(), "Variant<type: TensorList value: ?>");
  const TensorList& stored_vec = *x.get<TensorList>();
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(stored_vec.vec[i].flat<int>()(0), i);
  }
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(stored_vec.vec[i + 4].flat<float>()(0), 2 * i);
  }

  VariantTensorData serialized;
  x.Encode(&serialized);

  Variant y = TensorList();
  y.Decode(serialized);

  const TensorList& decoded_vec = *y.get<TensorList>();
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(decoded_vec.vec[i].flat<int>()(0), i);
  }
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(decoded_vec.vec[i + 4].flat<float>()(0), 2 * i);
  }

  VariantTensorDataProto data;
  serialized.ToProto(&data);
  const Variant y_unknown = data;
  EXPECT_EQ(y_unknown.TypeName(), "TensorList");
  EXPECT_EQ(y_unknown.TypeId(), TypeIndex::Make<VariantTensorDataProto>());
  EXPECT_EQ(y_unknown.DebugString(),
            strings::StrCat(
                "Variant<type: TensorList value: ", data.DebugString(), ">"));
}

template <bool BIG>
void TestVariantArray() {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_39(mht_39_v, 885, "", "./tensorflow/core/framework/variant_test.cc", "TestVariantArray");

  Variant x[2];
  x[0] = Int<BIG>{2};
  x[1] = Float<BIG>{2.0f};

  EXPECT_EQ(x[0].get<Int<BIG>>()->value, 2);
  EXPECT_EQ(x[1].get<Float<BIG>>()->value, 2.0f);
}

TEST(VariantTest, VariantArray) { TestVariantArray<false>(); }

TEST(VariantTest, VariantArrayBig) { TestVariantArray<true>(); }

template <bool BIG>
void PodUpdateTest() {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_40(mht_40_v, 902, "", "./tensorflow/core/framework/variant_test.cc", "PodUpdateTest");

  struct Pod {
    int x;
    float y;
    char big[BIG ? 256 : 1];

    string TypeName() const {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_41(mht_41_v, 911, "", "./tensorflow/core/framework/variant_test.cc", "TypeName");
 return "POD"; }
  };

  Variant x = Pod{10, 20.f};
  EXPECT_NE(x.get<Pod>(), nullptr);
  EXPECT_EQ(x.TypeName(), "POD");
  EXPECT_EQ(x.DebugString(), "Variant<type: POD value: ?>");

  x.get<Pod>()->x += x.get<Pod>()->y;
  EXPECT_EQ(x.get<Pod>()->x, 30);
}

TEST(VariantTest, PodUpdate) { PodUpdateTest<false>(); }

TEST(VariantTest, PodUpdateBig) { PodUpdateTest<true>(); }

template <bool BIG>
void TestEncodeDecodePod() {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_42(mht_42_v, 931, "", "./tensorflow/core/framework/variant_test.cc", "TestEncodeDecodePod");

  struct Pod {
    int x;
    float y;
    char big[BIG ? 256 : 1];

    string TypeName() const {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_testDTcc mht_43(mht_43_v, 940, "", "./tensorflow/core/framework/variant_test.cc", "TypeName");
 return "POD"; }
  };

  Variant x;
  Pod p{10, 20.0f};
  x = p;

  VariantTensorData serialized;
  x.Encode(&serialized);

  Variant y = Pod{};
  y.Decode(serialized);

  EXPECT_EQ(p.x, y.get<Pod>()->x);
  EXPECT_EQ(p.y, y.get<Pod>()->y);
}

TEST(VariantTest, EncodeDecodePod) { TestEncodeDecodePod<false>(); }

TEST(VariantTest, EncodeDecodePodBig) { TestEncodeDecodePod<true>(); }

TEST(VariantTest, EncodeDecodeTensor) {
  Variant x;
  Tensor t(DT_INT32, {});
  t.flat<int>()(0) = 42;
  x = t;

  VariantTensorData serialized;
  x.Encode(&serialized);

  Variant y = Tensor();
  y.Decode(serialized);
  EXPECT_EQ(y.DebugString(),
            "Variant<type: tensorflow::Tensor value: Tensor<type: int32 shape: "
            "[] values: 42>>");
  EXPECT_EQ(x.get<Tensor>()->flat<int>()(0), y.get<Tensor>()->flat<int>()(0));
}

}  // end namespace tensorflow
