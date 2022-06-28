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

// Implement the Philox algorithm to generate random numbers in parallel.
// Salmon et al. SC 2011. Parallel random numbers: as easy as 1, 2, 3.
//   http://www.thesalmons.org/john/random123/papers/random123sc11.pdf

#ifndef TENSORFLOW_CORE_LIB_RANDOM_PHILOX_RANDOM_H_
#define TENSORFLOW_CORE_LIB_RANDOM_PHILOX_RANDOM_H_
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
class MHTracer_DTPStensorflowPScorePSlibPSrandomPSphilox_randomDTh {
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
   MHTracer_DTPStensorflowPScorePSlibPSrandomPSphilox_randomDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSrandomPSphilox_randomDTh() {
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


#include <stdlib.h>

#include <cstdint>

// Function qualifiers that need to work on both CPU and GPU.
#if defined(__CUDACC__) || defined(__HIPCC__)
// For nvcc.
#define PHILOX_DEVICE_FUNC __host__ __device__
#define PHILOX_INLINE __inline__
#else
// For non-nvcc.
#define PHILOX_DEVICE_FUNC
#define PHILOX_INLINE inline
#endif
#define PHILOX_DEVICE_INLINE PHILOX_DEVICE_FUNC PHILOX_INLINE

#include <math.h>

namespace tensorflow {
namespace random {

// A class that represents an inline array. It can be used on both CPU and GPU,
// and also trivially copyable between CPU and GPU.
// Arguments:
//   T: the array element type;
//   ElementCount: the fixed size of the array;
template <typename T, int ElementCount>
class Array {
 public:
  static constexpr int kElementCount = ElementCount;
  PHILOX_DEVICE_INLINE Array() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSrandomPSphilox_randomDTh mht_0(mht_0_v, 222, "", "./tensorflow/core/lib/random/philox_random.h", "Array");

    for (int i = 0; i < ElementCount; ++i) {
      data_[i] = T(0);
    }
  }

  PHILOX_DEVICE_INLINE const T& operator[](int index) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSrandomPSphilox_randomDTh mht_1(mht_1_v, 231, "", "./tensorflow/core/lib/random/philox_random.h", "lambda");

    return data_[index];
  }

  PHILOX_DEVICE_INLINE T& operator[](int index) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSrandomPSphilox_randomDTh mht_2(mht_2_v, 238, "", "./tensorflow/core/lib/random/philox_random.h", "lambda");
 return data_[index]; }

  size_t size() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSrandomPSphilox_randomDTh mht_3(mht_3_v, 243, "", "./tensorflow/core/lib/random/philox_random.h", "size");
 return ElementCount; }

 private:
  T data_[ElementCount];
};

// A class that encapsulates all the states for a random number generator using
// the philox_4x32_10 algorithm. Each invocation returns a 128-bit random bits
// in the form of four uint32_t.
// There are multiple variants of this algorithm, we picked the 4x32_10 version
// that is most suited for our applications.
// Since this class is meant to be copied between CPU to GPU, it maintains a
// value semantics.
//
// For example: To use this class and populate an array of 1024 randoms on CPU
// with two threads,
//
//  void Fill(PhiloxRandom rnd, uint32_t* output, int start, int limit) {
//    assert(start % 4 == 0);
//    assert(limit % 4 == 0);
//    rnd.Skip(start / 4);
//    for (int i = start; i < limit; i += 4) {
//      auto sample = rnd();
//      ... copy sample[0..3] to output[i..i+3]
//    }
//  }
//
//  PhiloxRandom rng(seed);
//  PhiloxRandom rng_copy = rng;
//  rng.Skip(1000/4);
//
//  ... schedule Fill(rng_copy, output, 0, 512) in thread 1;
//  ... schedule Fill(rng_copy, output, 512, 1024) in thread 2;
//  ... wait for thread 1 & 2 to finish executing Fill().
//
// NOTE:
// 1. PhiloxRandom is trivially copyable.
// 2. PhiloxRandom is compilable by gcc and nvcc.
class PhiloxRandom {
 public:
  using ResultType = Array<uint32_t, 4>;
  using ResultElementType = uint32_t;
  // The number of elements that will be returned.
  static constexpr int kResultElementCount = 4;
  // Cost of generation of a single element (in cycles).
  static constexpr int kElementCost = 10;
  // The type for the 64-bit key stored in the form of two 32-bit uint
  // that are used in the diffusion process.
  using Key = Array<uint32_t, 2>;

  PHILOX_DEVICE_INLINE
  PhiloxRandom() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPSrandomPSphilox_randomDTh mht_4(mht_4_v, 297, "", "./tensorflow/core/lib/random/philox_random.h", "PhiloxRandom");
}

  PHILOX_DEVICE_INLINE
  explicit PhiloxRandom(uint64_t seed) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSlibPSrandomPSphilox_randomDTh mht_5(mht_5_v, 303, "", "./tensorflow/core/lib/random/philox_random.h", "PhiloxRandom");

    key_[0] = static_cast<uint32_t>(seed);
    key_[1] = static_cast<uint32_t>(seed >> 32);
  }

  PHILOX_DEVICE_INLINE
  explicit PhiloxRandom(uint64_t seed_lo, uint64_t seed_hi) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPSrandomPSphilox_randomDTh mht_6(mht_6_v, 312, "", "./tensorflow/core/lib/random/philox_random.h", "PhiloxRandom");

    key_[0] = static_cast<uint32_t>(seed_lo);
    key_[1] = static_cast<uint32_t>(seed_lo >> 32);
    counter_[2] = static_cast<uint32_t>(seed_hi);
    counter_[3] = static_cast<uint32_t>(seed_hi >> 32);
  }

  PHILOX_DEVICE_INLINE
  PhiloxRandom(ResultType counter, Key key) : counter_(counter), key_(key) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSlibPSrandomPSphilox_randomDTh mht_7(mht_7_v, 323, "", "./tensorflow/core/lib/random/philox_random.h", "PhiloxRandom");
}

  PHILOX_DEVICE_INLINE
  ResultType const& counter() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSlibPSrandomPSphilox_randomDTh mht_8(mht_8_v, 329, "", "./tensorflow/core/lib/random/philox_random.h", "counter");
 return counter_; }

  PHILOX_DEVICE_INLINE
  Key const& key() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSlibPSrandomPSphilox_randomDTh mht_9(mht_9_v, 335, "", "./tensorflow/core/lib/random/philox_random.h", "key");
 return key_; }

  // Skip the specified number of samples of 128-bits in the current stream.
  PHILOX_DEVICE_INLINE
  void Skip(uint64_t count) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSlibPSrandomPSphilox_randomDTh mht_10(mht_10_v, 342, "", "./tensorflow/core/lib/random/philox_random.h", "Skip");

    const uint32_t count_lo = static_cast<uint32_t>(count);
    uint32_t count_hi = static_cast<uint32_t>(count >> 32);

    counter_[0] += count_lo;
    if (counter_[0] < count_lo) {
      ++count_hi;
    }

    counter_[1] += count_hi;
    if (counter_[1] < count_hi) {
      if (++counter_[2] == 0) {
        ++counter_[3];
      }
    }
  }

  // Returns a group of four random numbers using the underlying Philox
  // algorithm.
  PHILOX_DEVICE_INLINE ResultType operator()() {
    ResultType counter = counter_;
    Key key = key_;

    // Run the single rounds for ten times. Manually unrolling the loop
    // for better performance.
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);

    SkipOne();

    return counter;
  }

 private:
  // We use the same constants as recommended by the original paper.
  static constexpr uint32_t kPhiloxW32A = 0x9E3779B9;
  static constexpr uint32_t kPhiloxW32B = 0xBB67AE85;
  static constexpr uint32_t kPhiloxM4x32A = 0xD2511F53;
  static constexpr uint32_t kPhiloxM4x32B = 0xCD9E8D57;

  // Helper function to skip the next sample of 128-bits in the current stream.
  PHILOX_DEVICE_INLINE void SkipOne() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSlibPSrandomPSphilox_randomDTh mht_11(mht_11_v, 403, "", "./tensorflow/core/lib/random/philox_random.h", "SkipOne");

    if (++counter_[0] == 0) {
      if (++counter_[1] == 0) {
        if (++counter_[2] == 0) {
          ++counter_[3];
        }
      }
    }
  }

  // Helper function to return the lower and higher 32-bits from two 32-bit
  // integer multiplications.
  PHILOX_DEVICE_INLINE
  static void MultiplyHighLow(uint32_t a, uint32_t b, uint32_t* result_low,
                              uint32_t* result_high) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSlibPSrandomPSphilox_randomDTh mht_12(mht_12_v, 420, "", "./tensorflow/core/lib/random/philox_random.h", "MultiplyHighLow");

#ifndef __CUDA_ARCH__
    const uint64_t product = static_cast<uint64_t>(a) * b;
    *result_low = static_cast<uint32_t>(product);
    *result_high = static_cast<uint32_t>(product >> 32);
#else
    *result_low = a * b;
    *result_high = __umulhi(a, b);
#endif
  }

  // Helper function for a single round of the underlying Philox algorithm.
  PHILOX_DEVICE_INLINE static ResultType ComputeSingleRound(
      const ResultType& counter, const Key& key) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSlibPSrandomPSphilox_randomDTh mht_13(mht_13_v, 436, "", "./tensorflow/core/lib/random/philox_random.h", "ComputeSingleRound");

    uint32_t lo0;
    uint32_t hi0;
    MultiplyHighLow(kPhiloxM4x32A, counter[0], &lo0, &hi0);

    uint32_t lo1;
    uint32_t hi1;
    MultiplyHighLow(kPhiloxM4x32B, counter[2], &lo1, &hi1);

    ResultType result;
    result[0] = hi1 ^ counter[1] ^ key[0];
    result[1] = lo1;
    result[2] = hi0 ^ counter[3] ^ key[1];
    result[3] = lo0;
    return result;
  }

  PHILOX_DEVICE_INLINE void RaiseKey(Key* key) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSlibPSrandomPSphilox_randomDTh mht_14(mht_14_v, 456, "", "./tensorflow/core/lib/random/philox_random.h", "RaiseKey");

    (*key)[0] += kPhiloxW32A;
    (*key)[1] += kPhiloxW32B;
  }

 private:
  ResultType counter_;
  Key key_;
};

}  // namespace random
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_RANDOM_PHILOX_RANDOM_H_
