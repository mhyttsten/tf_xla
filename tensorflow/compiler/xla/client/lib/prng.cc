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
class MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSprngDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSprngDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSprngDTcc() {
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

#include "tensorflow/compiler/xla/client/lib/prng.h"

#include <cmath>
#include <vector>

#include "absl/base/casts.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

xla::XlaOp ConcatScalars(xla::XlaBuilder* builder,
                         absl::Span<const xla::XlaOp> scalars) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSprngDTcc mht_0(mht_0_v, 199, "", "./tensorflow/compiler/xla/client/lib/prng.cc", "ConcatScalars");

  std::vector<xla::XlaOp> vectors;
  absl::c_transform(scalars, std::back_inserter(vectors),
                    [](xla::XlaOp x) { return xla::Reshape(x, {1}); });
  return ConcatInDim(builder, vectors, 0);
}

namespace {

// Rotates a 32-bit integer 'v' left by 'distance' bits.
XlaOp RotateLeftU32(XlaOp v, int distance) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSprngDTcc mht_1(mht_1_v, 212, "", "./tensorflow/compiler/xla/client/lib/prng.cc", "RotateLeftU32");

  return (v << ConstantR0<uint32_t>(v.builder(), distance)) |
         ShiftRightLogical(v, ConstantR0<uint32_t>(v.builder(), 32 - distance));
}

// The internal state of the Three Fry implementation.
using ThreeFry2x32State = std::array<XlaOp, 2>;

// Implements the ThreeFry counter-based PRNG algorithm.
// Salmon et al. SC 2011. Parallel random numbers: as easy as 1, 2, 3.
// http://www.thesalmons.org/john/random123/papers/random123sc11.pdf
ThreeFry2x32State ThreeFry2x32(ThreeFry2x32State input, ThreeFry2x32State key) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSprngDTcc mht_2(mht_2_v, 226, "", "./tensorflow/compiler/xla/client/lib/prng.cc", "ThreeFry2x32");

  XlaBuilder* builder = input[0].builder();
  key[0] = BitcastConvertType(key[0], U32);
  key[1] = BitcastConvertType(key[1], U32);

  // Rotation distances specified by the Threefry2x32 algorithm.
  constexpr std::array<int, 8> rotations = {13, 15, 26, 6, 17, 29, 16, 24};
  ThreeFry2x32State x;

  std::array<XlaOp, 3> ks;
  // 0x1BD11BDA is a parity constant specified by the ThreeFry2x32 algorithm.
  ks[2] = ConstantR0<uint32_t>(builder, 0x1BD11BDA);
  for (int i = 0; i < 2; ++i) {
    ks[i] = key[i];
    x[i] = input[i];
    ks[2] = ks[2] ^ key[i];
  }

  x[0] = x[0] + ks[0];
  x[1] = x[1] + ks[1];

  // Performs a single round of the Threefry2x32 algorithm, with a rotation
  // amount 'rotation'.
  auto round = [](ThreeFry2x32State v, int rotation) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSprngDTcc mht_3(mht_3_v, 252, "", "./tensorflow/compiler/xla/client/lib/prng.cc", "lambda");

    v[0] = v[0] + v[1];
    v[1] = RotateLeftU32(v[1], rotation);
    v[1] = v[0] ^ v[1];
    return v;
  };

  // There are no known statistical flaws with 13 rounds of Threefry2x32.
  // We are conservative and use 20 rounds.
  x = round(x, rotations[0]);
  x = round(x, rotations[1]);
  x = round(x, rotations[2]);
  x = round(x, rotations[3]);
  x[0] = x[0] + ks[1];
  x[1] = x[1] + ks[2] + ConstantR0<uint32_t>(builder, 1);

  x = round(x, rotations[4]);
  x = round(x, rotations[5]);
  x = round(x, rotations[6]);
  x = round(x, rotations[7]);
  x[0] = x[0] + ks[2];
  x[1] = x[1] + ks[0] + ConstantR0<uint32_t>(builder, 2);

  x = round(x, rotations[0]);
  x = round(x, rotations[1]);
  x = round(x, rotations[2]);
  x = round(x, rotations[3]);
  x[0] = x[0] + ks[0];
  x[1] = x[1] + ks[1] + ConstantR0<uint32_t>(builder, 3);

  x = round(x, rotations[4]);
  x = round(x, rotations[5]);
  x = round(x, rotations[6]);
  x = round(x, rotations[7]);
  x[0] = x[0] + ks[1];
  x[1] = x[1] + ks[2] + ConstantR0<uint32_t>(builder, 4);

  x = round(x, rotations[0]);
  x = round(x, rotations[1]);
  x = round(x, rotations[2]);
  x = round(x, rotations[3]);
  x[0] = x[0] + ks[2];
  x[1] = x[1] + ks[0] + ConstantR0<uint32_t>(builder, 5);

  return x;
}

// Converts a uint64_t to two uint32s.
std::array<XlaOp, 2> Uint64ToUint32s(XlaOp u64) {
  XlaBuilder* builder = u64.builder();
  XlaOp const32 = ConstantR0WithType(builder, U64, 32);
  XlaOp fst = ConvertElementType(u64, U32);
  XlaOp snd = ConvertElementType(ShiftRightLogical(u64, const32), U32);
  return {fst, snd};
}

// Converts two uint32s to a uint64_t.
XlaOp Uint32sToUint64(std::array<XlaOp, 2> u32s) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSprngDTcc mht_4(mht_4_v, 312, "", "./tensorflow/compiler/xla/client/lib/prng.cc", "Uint32sToUint64");

  XlaBuilder* builder = u32s[0].builder();
  return ConvertElementType(u32s[0], U64) |
         ShiftLeft(ConvertElementType(u32s[1], U64),
                   ConstantR0WithType(builder, U64, 32));
}

// Given the initial state and the request shape of random numbers to be
// generated, returns the input for the random number generator and a new state.
std::pair<ThreeFry2x32State, XlaOp> GetThreeFryInputsAndUpdatedState(
    XlaOp initial_state, const Shape& shape) {
  XlaBuilder* builder = initial_state.builder();
  auto u64_shape = ShapeUtil::MakeShape(U64, shape.dimensions());
  // initial_state is an R1, so reshape it to a scalar.
  auto input_u64 = Broadcast(Reshape(initial_state, {}), shape.dimensions());
  int64_t trailing_dims_product = 1;
  for (int64_t i = shape.rank() - 1; i >= 0; --i) {
    if (shape.dimensions(i) < 2) {
      continue;
    }
    input_u64 =
        input_u64 + (Iota(builder, u64_shape, i) *
                     ConstantR0<uint64_t>(builder, trailing_dims_product));
    trailing_dims_product *= shape.dimensions(i);
  }
  XlaOp new_state = initial_state +
                    ConstantR0<uint64_t>(builder, ShapeUtil::ElementsIn(shape));
  return std::make_pair(Uint64ToUint32s(input_u64), new_state);
}

// Result for SplitShapeIntoHalves().
struct SplitShapePair {
  Shape half_shape;
  Shape concat_shape;
  int64_t split_dim;
  int64_t new_concat_dim;
};

// Split the shape on a dimension > 1 into two halves.
SplitShapePair SplitShapeIntoHalves(const Shape& shape) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSprngDTcc mht_5(mht_5_v, 354, "", "./tensorflow/compiler/xla/client/lib/prng.cc", "SplitShapeIntoHalves");

  SplitShapePair pair;
  if (shape.rank() == 0) {
    pair.half_shape = ShapeUtil::MakeShape(shape.element_type(), {1});
    pair.concat_shape = ShapeUtil::MakeShape(shape.element_type(), {2});
    pair.split_dim = 0;
    pair.new_concat_dim = 0;
    return pair;
  }
  pair.split_dim = -1;
  for (int64_t i = 0; i < shape.rank(); ++i) {
    if (shape.dimensions(i) % 2 == 0) {
      pair.split_dim = i;
      break;
    }
  }
  if (pair.split_dim == -1) {
    // No even dims. Find a dimension with maximum size.
    for (int64_t i = 0; i < shape.rank(); ++i) {
      if (pair.split_dim == -1 ||
          shape.dimensions(i) > shape.dimensions(pair.split_dim)) {
        pair.split_dim = i;
      }
    }
  }
  CHECK_GE(pair.split_dim, 0);
  std::vector<int64_t> half_shape_dims;
  std::vector<int64_t> concat_shape_dims;
  const auto rank = shape.rank();
  half_shape_dims.reserve(rank + 1);
  concat_shape_dims.reserve(rank + 1);
  for (int64_t i = 0; i < rank; ++i) {
    if (i == pair.split_dim) {
      // Create a new trivial dim for the later concat, which is more friendly
      // to sharding propagation.
      half_shape_dims.push_back(CeilOfRatio<int64_t>(shape.dimensions(i), 2));
      half_shape_dims.push_back(1);
      concat_shape_dims.push_back(half_shape_dims[i]);
      concat_shape_dims.push_back(2);
    } else {
      half_shape_dims.push_back(shape.dimensions(i));
      concat_shape_dims.push_back(shape.dimensions(i));
    }
  }
  pair.new_concat_dim = pair.split_dim + 1;
  pair.half_shape = ShapeUtil::MakeShape(shape.element_type(), half_shape_dims);
  pair.concat_shape =
      ShapeUtil::MakeShape(shape.element_type(), concat_shape_dims);
  return pair;
}

// Combines a pair of split shapes. It works with scalar and non-scalar shapes.
XlaOp CombineShapePair(absl::Span<const XlaOp> pair,
                       const SplitShapePair& shape_pair,
                       const Shape& original_shape) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSprngDTcc mht_6(mht_6_v, 411, "", "./tensorflow/compiler/xla/client/lib/prng.cc", "CombineShapePair");

  if (original_shape.rank() == 0) {
    return Reshape(pair[0], {});
  }
  XlaBuilder* builder = pair[0].builder();
  XlaOp result = ConcatInDim(builder, pair, shape_pair.new_concat_dim);
  const int64_t pre_split_size =
      original_shape.dimensions(shape_pair.split_dim);
  std::vector<int64_t> reshape_dims(original_shape.dimensions().begin(),
                                    original_shape.dimensions().end());
  reshape_dims[shape_pair.split_dim] = RoundUpTo<int64_t>(pre_split_size, 2);
  result = Reshape(result, reshape_dims);
  if (reshape_dims[shape_pair.split_dim] != pre_split_size) {
    result = Slice(result, std::vector<int64_t>(original_shape.rank(), 0),
                   original_shape.dimensions(),
                   std::vector<int64_t>(original_shape.rank(), 1));
  }
  return result;
}

// Generates random 32bits with the given shape using the Three Fry
// implementation. Returns the random bits and the new state.
RngOutput ThreeFryRngBit32(XlaOp key, XlaOp initial_state, const Shape& shape) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSprngDTcc mht_7(mht_7_v, 436, "", "./tensorflow/compiler/xla/client/lib/prng.cc", "ThreeFryRngBit32");

  auto shape_pair = SplitShapeIntoHalves(shape);
  std::pair<ThreeFry2x32State, XlaOp> inputs_state =
      GetThreeFryInputsAndUpdatedState(initial_state, shape_pair.half_shape);
  ThreeFry2x32State inputs = inputs_state.first;
  ThreeFry2x32State outputs = ThreeFry2x32(inputs, Uint64ToUint32s(key));
  XlaOp result = CombineShapePair(outputs, shape_pair, shape);
  return {result, inputs_state.second};
}

// Generates random 64bits with the given shape using the Three Fry
// implementation. Returns the random bits and the new state.
RngOutput ThreeFryRngBit64(XlaOp key, XlaOp initial_state, const Shape& shape) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSprngDTcc mht_8(mht_8_v, 451, "", "./tensorflow/compiler/xla/client/lib/prng.cc", "ThreeFryRngBit64");

  std::pair<ThreeFry2x32State, XlaOp> inputs_state =
      GetThreeFryInputsAndUpdatedState(initial_state, shape);
  ThreeFry2x32State inputs = inputs_state.first;
  ThreeFry2x32State outputs = ThreeFry2x32(inputs, Uint64ToUint32s(key));
  XlaOp result = Uint32sToUint64(outputs);
  return {result, inputs_state.second};
}

// The key of the Philox random number generator.
using Philox4x32Key = std::array<XlaOp, 2>;
// The internal state of the Philox random number generator.
using Philox4x32State = std::array<XlaOp, 4>;

// Computes the Philox4x32 algorithm using 10 rounds.
Philox4x32State Philox4x32(Philox4x32State state, Philox4x32Key key) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSprngDTcc mht_9(mht_9_v, 469, "", "./tensorflow/compiler/xla/client/lib/prng.cc", "Philox4x32");

  // Constants specified by the Philox algorithm.
  static const uint32_t kPhiloxW32A = 0x9E3779B9;
  static const uint32_t kPhiloxW32B = 0xBB67AE85;
  static const uint32_t kPhiloxM4x32A = 0xD2511F53;
  static const uint32_t kPhiloxM4x32B = 0xCD9E8D57;

  struct HighLowPair {
    XlaOp high;
    XlaOp low;
  };

  // Compute the high and low words from multiplying two 32-bit integers.
  auto mul_hi_low = [](XlaOp x, uint32_t k) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSprngDTcc mht_10(mht_10_v, 485, "", "./tensorflow/compiler/xla/client/lib/prng.cc", "lambda");

    auto product =
        ConvertElementType(x, U64) * ConstantR0<uint64_t>(x.builder(), k);
    auto low = ConvertElementType(product, U32);
    auto high = ConvertElementType(
        product >> ConstantR0<uint64_t>(x.builder(), 32), U32);
    return HighLowPair{high, low};
  };

  // Perform a single round of the Philox algorithm.
  auto philox_round = [&](Philox4x32State x, Philox4x32Key key) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSprngDTcc mht_11(mht_11_v, 498, "", "./tensorflow/compiler/xla/client/lib/prng.cc", "lambda");

    auto product0 = mul_hi_low(x[0], kPhiloxM4x32A);
    auto product1 = mul_hi_low(x[2], kPhiloxM4x32B);
    return Philox4x32State{product1.high ^ x[1] ^ key[0], product1.low,
                           product0.high ^ x[3] ^ key[1], product0.low};
  };

  // Update the key after a round of Philox algorithm.
  auto raise_key = [](Philox4x32Key key) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSprngDTcc mht_12(mht_12_v, 509, "", "./tensorflow/compiler/xla/client/lib/prng.cc", "lambda");

    XlaBuilder* builder = key[0].builder();
    return Philox4x32Key{key[0] + ConstantR0<uint32_t>(builder, kPhiloxW32A),
                         key[1] + ConstantR0<uint32_t>(builder, kPhiloxW32B)};
  };

  static const int kNumRounds = 10;
  for (int round = 0; round < kNumRounds; ++round, key = raise_key(key)) {
    state = philox_round(state, key);
  }
  return state;
}

// Scrambles the input key so that users don't need to worry about which part
// of the key needs to be strong.
std::pair<Philox4x32State, Philox4x32Key> ScramblePhiloxKey(Philox4x32Key key) {
  XlaBuilder* builder = key[0].builder();
  XlaOp key0 = ConvertElementType(key[0], U64);
  XlaOp key1 = ConvertElementType(key[1], U64);

  Philox4x32State state = {
      ConvertElementType(key0, U32),
      ConvertElementType(key0 >> ScalarLike(key0, 32), U32),
      ConvertElementType(key1, U32),
      ConvertElementType(key1 >> ScalarLike(key1, 32), U32),
  };
  key = {ConstantR0<uint32_t>(builder, 0x3ec8f720),
         ConstantR0<uint32_t>(builder, 0x02461e29)};
  state = Philox4x32(state, key);
  XlaOp zero = ConstantR0<uint32_t>(builder, 0);
  return {Philox4x32State{zero, zero, state[2], state[3]},
          Philox4x32Key{state[0], state[1]}};
}

// Adds an U128 tensor with an U64 tensor. The U128 tensor is represented as two
// U64s with the low 64bits in the front. This routine supports explicit
// broadcasting of the U128 tensor, with `broadcast_sizes` representing the
// dimensions prepended to its shape.
std::array<XlaOp, 2> Uint128AddUint64(
    const std::array<XlaOp, 2>& u128, XlaOp u64,
    absl::Span<const int64_t> broadcast_sizes = {}) {
  auto u128_low = u128[0];
  auto u128_high = u128[1];
  XlaOp new_u128_low = u128_low + u64;
  XlaOp one = ConstantR0<uint64_t>(u128[0].builder(), 1);
  XlaOp new_u128_high = Select(Lt(new_u128_low, u128_low),
                               Broadcast(u128_high + one, broadcast_sizes),
                               Broadcast(u128_high, broadcast_sizes));
  return {new_u128_low, new_u128_high};
}

std::array<XlaOp, 2> Uint32sToUint128(const std::array<XlaOp, 4>& u32s) {
  return {Uint32sToUint64({u32s[0], u32s[1]}),
          Uint32sToUint64({u32s[2], u32s[3]})};
}

std::array<XlaOp, 4> Uint128ToUint32s(const std::array<XlaOp, 2>& u128) {
  std::array<XlaOp, 2> u128_low_32s = Uint64ToUint32s(u128[0]);
  std::array<XlaOp, 2> u128_high_32s = Uint64ToUint32s(u128[1]);
  return {u128_low_32s[0], u128_low_32s[1], u128_high_32s[0], u128_high_32s[1]};
}

std::array<XlaOp, 2> Uint128FromOp(XlaOp op) {
  auto u128_low = xla::Reshape(xla::Slice(op, {0}, {1}, {1}), {});
  auto u128_high = xla::Reshape(xla::Slice(op, {1}, {2}, {1}), {});
  return {u128_low, u128_high};
}

XlaOp Uint128ToOp(std::array<XlaOp, 2> u128) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSprngDTcc mht_13(mht_13_v, 580, "", "./tensorflow/compiler/xla/client/lib/prng.cc", "Uint128ToOp");

  return ConcatScalars(u128[0].builder(), {u128[0], u128[1]});
}

// Returns the pair (state + [0, 1, ..., n-1], state + n), which should be used
// as the inputs fed to `Philox4x32` and the updated state. `state` is an U128
// represented as 4 U32s in the order from the least significant one to the most
// significant one.
std::pair<Philox4x32State, XlaOp> GetPhiloxInputsAndUpdatedState(
    const Philox4x32State& state, int64_t n) {
  XlaBuilder* builder = state[0].builder();
  XlaOp iota = Iota(builder, U64, n);
  auto state_u128 = Uint32sToUint128(state);
  auto inputs = Uint128ToUint32s(Uint128AddUint64(state_u128, iota, {n}));
  XlaOp new_state = Uint128ToOp(
      Uint128AddUint64(state_u128, ConstantR0<uint64_t>(builder, n)));
  return std::make_pair(inputs, new_state);
}

// Generates CeilOfRatio(num_elems, 4)*4 32bit Philox random numbers, as Philox
// numbers are generated in the unit of 128bits.
std::pair<Philox4x32State, XlaOp> GeneratePhiloxBits(int64_t num_elems,
                                                     XlaOp initial_state,
                                                     Philox4x32Key key) {
  Philox4x32State state;
  state = Uint128ToUint32s(Uint128FromOp(initial_state));
  const int64_t num_vector4 = CeilOfRatio<int64_t>(num_elems, 4);
  Philox4x32State inputs;
  XlaOp new_state;
  std::tie(inputs, new_state) =
      GetPhiloxInputsAndUpdatedState(state, num_vector4);
  auto outputs = Philox4x32(inputs, key);
  return std::make_pair(outputs, new_state);
}

// Generates an array of primitive type U32 with the given shape containing
// random bits generated by the Philox algorithm. Returns the array and the new
// state of the random number generator.
RngOutput PhiloxRngBit32(XlaOp op_key, XlaOp initial_state,
                         const Shape& shape) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSprngDTcc mht_14(mht_14_v, 622, "", "./tensorflow/compiler/xla/client/lib/prng.cc", "PhiloxRngBit32");

  XlaBuilder* builder = op_key.builder();
  const int64_t num_elems = ShapeUtil::ElementsIn(shape);

  Philox4x32Key key = Uint64ToUint32s(op_key);
  Philox4x32State bits;
  XlaOp new_state;
  std::tie(bits, new_state) = GeneratePhiloxBits(num_elems, initial_state, key);
  // Combining bits[i] in a round-robin fashion, to align with non-XLA
  // implementations
  int64_t bits_len = (num_elems + 3) / 4;
  for (auto i = 0; i < 4; ++i) {
    bits[i] = Reshape(bits[i], {bits_len, 1});
  }
  XlaOp numbers = ConcatInDim(builder, {bits[0], bits[1], bits[2], bits[3]},
                              /*dimension=*/1);
  numbers = Reshape(numbers, {bits_len * 4});
  numbers = Slice(numbers, /*start_indices=*/{0},
                  /*limit_indices=*/{num_elems},
                  /*strides=*/{1});
  return {Reshape(numbers, shape.dimensions()), new_state};
}

// Generates an array of primitive type U64 with the given shape containing
// random bits generated by the Philox algorithm. Returns the array and the new
// state of the random number generator.
RngOutput PhiloxRngBit64(XlaOp op_key, XlaOp initial_state,
                         const Shape& shape) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSprngDTcc mht_15(mht_15_v, 652, "", "./tensorflow/compiler/xla/client/lib/prng.cc", "PhiloxRngBit64");

  XlaBuilder* builder = op_key.builder();
  const int64_t num_elems = ShapeUtil::ElementsIn(shape);

  Philox4x32Key key = Uint64ToUint32s(op_key);
  Philox4x32State bits32;
  XlaOp new_state;
  std::tie(bits32, new_state) =
      GeneratePhiloxBits(num_elems * 2, initial_state, key);

  std::array<XlaOp, 2> bits64;
  bits64[0] = Uint32sToUint64({bits32[0], bits32[1]});
  bits64[1] = Uint32sToUint64({bits32[2], bits32[3]});

  // Combining bits64[i] in a round-robin fashion, to align with non-XLA
  // implementations
  int64_t bits64_len = (num_elems + 1) / 2;
  for (auto i = 0; i < 2; ++i) {
    bits64[i] = Reshape(bits64[i], {bits64_len, 1});
  }
  XlaOp numbers = ConcatInDim(builder, {bits64[0], bits64[1]},
                              /*dimension=*/1);
  numbers = Reshape(numbers, {bits64_len * 2});
  numbers = Slice(numbers, /*start_indices=*/{0},
                  /*limit_indices=*/{num_elems},
                  /*strides=*/{1});
  return {Reshape(numbers, shape.dimensions()), new_state};
}

XlaOp ConvertRandomBitsToUniformFloatingPoint(XlaOp bits, XlaOp minval,
                                              XlaOp maxval) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSprngDTcc mht_16(mht_16_v, 685, "", "./tensorflow/compiler/xla/client/lib/prng.cc", "ConvertRandomBitsToUniformFloatingPoint");

  XlaBuilder* builder = bits.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape* minval_shape,
                        builder->GetShapePtr(minval));
    TF_ASSIGN_OR_RETURN(const Shape* bits_shape, builder->GetShapePtr(bits));
    PrimitiveType value_type = minval_shape->element_type();
    PrimitiveType bit_type = bits_shape->element_type();
    CHECK((value_type == F32 && bit_type == U32) ||
          (value_type == F64 && bit_type == U64));

    // Form random mantissa bits for float/double, with a leading 1 bit.
    int num_float_bits = primitive_util::BitWidth(value_type);
    // Subtract one as SignificandWidth includes the leading 1 bit.
    int num_mantissa_bits = primitive_util::SignificandWidth(value_type) - 1;

    // Ignore the exponent bits and convert the mantissa bits to the floating
    // point type.
    bits = ShiftRightLogical(
        bits, ScalarLike(bits, num_float_bits - num_mantissa_bits));

    // We have an integer-valued floating point number in the range
    // [0, 2**{num_mantissa_bits}).
    XlaOp values = ConvertElementType(bits, value_type);

    // Divide by 2**{-num_mantissa_bits} to get a number in the range
    // [0.0, 1.0).
    values = values * ScalarLike(values, std::ldexp(1., -num_mantissa_bits));

    // Multiply and add to shift to the range [minval, maxval).
    return values * (maxval - minval) + minval;
  });
}

XlaOp ConvertRandomBitsToUniformInt(XlaOp bits, XlaOp minval, XlaOp maxval,
                                    PrimitiveType type,
                                    PrimitiveType unsigned_type) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSprngDTcc mht_17(mht_17_v, 724, "", "./tensorflow/compiler/xla/client/lib/prng.cc", "ConvertRandomBitsToUniformInt");

  XlaBuilder* builder = bits.builder();
  XlaOp range = BitcastConvertType(maxval, unsigned_type) -
                BitcastConvertType(minval, unsigned_type);
  XlaOp dist = Rem(bits, range);
  XlaOp dist_div_2 =
      ShiftRightLogical(dist, ConstantR0WithType(builder, unsigned_type, 1));

  return minval + BitcastConvertType(dist_div_2, type) +
         BitcastConvertType(dist - dist_div_2, type);
}

// Implements the Box-Muller transform, which converts random floats in the
// range of [0, 1] from uniform distribution to normal distribution with mean 0
// and variance 1. For more detail on the Box-Muller transform, see
// http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform#Basic_form
std::pair<XlaOp, XlaOp> BoxMullerTransform(XlaOp x0, XlaOp x1) {
  // Do not send a really small number to log().
  XlaOp u1 = Max(x0, ScalarLike(x0, 1.0e-7f));

  XlaOp v1 = ScalarLike(x1, 2.0f * M_PI) * x1;
  XlaOp u2 = Sqrt(ScalarLike(u1, -2.0f) * Log(u1));
  return {Sin(v1) * u2, Cos(v1) * u2};
}

}  // namespace

XlaOp PhiloxIncreaseCounter(XlaOp counter, XlaOp delta) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSprngDTcc mht_18(mht_18_v, 754, "", "./tensorflow/compiler/xla/client/lib/prng.cc", "PhiloxIncreaseCounter");

  return Uint128ToOp(Uint128AddUint64(Uint128FromOp(counter), delta));
}

RngOutput ThreeFryBitGenerator(XlaOp key, XlaOp initial_state,
                               const Shape& shape) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSprngDTcc mht_19(mht_19_v, 762, "", "./tensorflow/compiler/xla/client/lib/prng.cc", "ThreeFryBitGenerator");

  PrimitiveType type = shape.element_type();
  switch (type) {
    case F32:
    case U32:
    case S32:
      return ThreeFryRngBit32(key, initial_state, shape);
    case F64:
    case U64:
    case S64:
      return ThreeFryRngBit64(key, initial_state, shape);
    default:
      return {key.builder()->ReportError(Unimplemented(
                  "Types other than F32, F64, U32, S32, U64 and S64 "
                  "are not implemented by ThreeFryBitGenerator; got %s",
                  primitive_util::LowercasePrimitiveTypeName(type))),
              initial_state};
  }
}

RngOutput PhiloxBitGenerator(XlaOp key, XlaOp initial_state,
                             const Shape& shape) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSprngDTcc mht_20(mht_20_v, 786, "", "./tensorflow/compiler/xla/client/lib/prng.cc", "PhiloxBitGenerator");

  PrimitiveType type = shape.element_type();
  switch (type) {
    case F32:
    case U32:
    case S32:
      return PhiloxRngBit32(key, initial_state, shape);
    case F64:
    case U64:
    case S64:
      return PhiloxRngBit64(key, initial_state, shape);
    default:
      return {key.builder()->ReportError(Unimplemented(
                  "Types other than F32, F64, U32, S32, U64 and S64 "
                  "are not implemented by PhiloxFryBitGenerator; got %s",
                  primitive_util::LowercasePrimitiveTypeName(type))),
              initial_state};
  }
}

std::pair<XlaOp, XlaOp> ScramblePhiloxKey(XlaOp key) {
  Philox4x32Key pkey = Uint64ToUint32s(key);
  auto state_key = ScramblePhiloxKey(pkey);
  return std::make_pair(Uint128ToOp(Uint32sToUint128(state_key.first)),
                        Uint32sToUint64(state_key.second));
}

RngOutput UniformFloatingPointDistribution(XlaOp key, XlaOp initial_state,
                                           BitGeneratorTy bit_generator,
                                           XlaOp minval, XlaOp maxval,
                                           const Shape& shape) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSprngDTcc mht_21(mht_21_v, 819, "", "./tensorflow/compiler/xla/client/lib/prng.cc", "UniformFloatingPointDistribution");

  RngOutput bits_state = bit_generator(key, initial_state, shape);
  XlaOp bits = bits_state.value;
  XlaOp new_state = bits_state.state;
  return {ConvertRandomBitsToUniformFloatingPoint(bits, minval, maxval),
          new_state};
}

RngOutput UniformIntDistribution(XlaOp key, XlaOp initial_state,
                                 BitGeneratorTy bit_generator, XlaOp minval,
                                 XlaOp maxval, const Shape& shape) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSprngDTcc mht_22(mht_22_v, 832, "", "./tensorflow/compiler/xla/client/lib/prng.cc", "UniformIntDistribution");

  RngOutput bits_state = bit_generator(key, initial_state, shape);
  XlaOp bits = bits_state.value;
  XlaOp new_state = bits_state.state;
  PrimitiveType type = shape.element_type();
  PrimitiveType unsigned_type;
  if (type == U32 || type == S32) {
    unsigned_type = U32;
  } else {
    DCHECK(type == U64 || type == S64);
    unsigned_type = U64;
  }
  return {
      ConvertRandomBitsToUniformInt(bits, minval, maxval, type, unsigned_type),
      new_state};
}

RngOutput NormalFloatingPointDistribution(XlaOp key, XlaOp initial_state,
                                          BitGeneratorTy bit_generator,
                                          const Shape& shape) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSprngDTcc mht_23(mht_23_v, 854, "", "./tensorflow/compiler/xla/client/lib/prng.cc", "NormalFloatingPointDistribution");

  PrimitiveType primitive_type = shape.element_type();
  DCHECK(primitive_type == F32 || primitive_type == F64);

  XlaBuilder* builder = key.builder();
  auto shape_pair = SplitShapeIntoHalves(shape);
  RngOutput bits_state = UniformFloatingPointDistribution(
      key, initial_state, bit_generator,
      xla::ConstantR0WithType(builder, primitive_type, 0.0),
      xla::ConstantR0WithType(builder, primitive_type, 1.0),
      shape_pair.concat_shape);

  // Separate the bits into two groups to perform the Box-Muller transform.
  XlaOp bits_0 = Slice(bits_state.value,
                       std::vector<int64_t>(shape_pair.half_shape.rank(), 0),
                       shape_pair.half_shape.dimensions(),
                       std::vector<int64_t>(shape_pair.half_shape.rank(), 1));
  std::vector<int64_t> bits_1_starts(shape_pair.half_shape.rank(), 0);
  bits_1_starts[shape_pair.new_concat_dim] = 1;
  XlaOp bits_1 = Slice(bits_state.value, bits_1_starts,
                       shape_pair.concat_shape.dimensions(),
                       std::vector<int64_t>(shape_pair.half_shape.rank(), 1));
  std::tie(bits_0, bits_1) = BoxMullerTransform(bits_0, bits_1);

  // Put the numbers in the two groups back to form the requested shape.
  XlaOp normal = CombineShapePair({bits_0, bits_1}, shape_pair, shape);
  return {normal, bits_state.state};
}

}  // namespace xla
