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
class MHTracer_DTPStensorflowPSlitePSobjcPSapisPSTFLTensorDTh {
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
   MHTracer_DTPStensorflowPSlitePSobjcPSapisPSTFLTensorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSobjcPSapisPSTFLTensorDTh() {
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

// Copyright 2018 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#import <Foundation/Foundation.h>

@class TFLQuantizationParameters;

NS_ASSUME_NONNULL_BEGIN

/**
 * @enum TFLTensorDataType
 * This enum specifies supported TensorFlow Lite tensor data types.
 */
typedef NS_ENUM(NSUInteger, TFLTensorDataType) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSobjcPSapisPSTFLTensorDTh mht_0(mht_0_v, 194, "", "./tensorflow/lite/objc/apis/TFLTensor.h", "NS_ENUM");

  /** Tensor data type not available. This indicates an error with the model. */
  TFLTensorDataTypeNoType,

  /** 32-bit single precision floating point. */
  TFLTensorDataTypeFloat32,

  /** 16-bit half precision floating point. */
  TFLTensorDataTypeFloat16,

  /** 32-bit signed integer. */
  TFLTensorDataTypeInt32,

  /** 8-bit unsigned integer. */
  TFLTensorDataTypeUInt8,

  /** 64-bit signed integer. */
  TFLTensorDataTypeInt64,

  /** Boolean. */
  TFLTensorDataTypeBool,

  /** 16-bit signed integer. */
  TFLTensorDataTypeInt16,

  /** 8-bit signed integer. */
  TFLTensorDataTypeInt8,

  /** 64-bit double precision floating point. */
  TFLTensorDataTypeFloat64,
};

/**
 * An input or output tensor in a TensorFlow Lite model.
 *
 * @warning Each `TFLTensor` instance is associated with a `TFLInterpreter` instance. Multiple
 *     `TFLTensor` instances of the same TensorFlow Lite model are associated with the same
 *     `TFLInterpreter` instance. As long as a `TFLTensor` instance is still in use, its associated
 *     `TFLInterpreter` instance will not be deallocated.
 */
@interface TFLTensor : NSObject

/** Name of the tensor. */
@property(nonatomic, readonly, copy) NSString *name;

/** Data type of the tensor. */
@property(nonatomic, readonly) TFLTensorDataType dataType;

/** Parameters for asymmetric quantization. `nil` if the tensor does not use quantization. */
@property(nonatomic, readonly, nullable) TFLQuantizationParameters *quantizationParameters;

/** Unavailable. */
- (instancetype)init NS_UNAVAILABLE;

/**
 * Copies the given data into an input tensor. This is allowed only for an input tensor and only
 * before the interpreter is invoked; otherwise an error will be returned.
 *
 * @param data The data to set. The byte size of the data must match what's required by the input
 *     tensor.
 * @param error An optional error parameter populated when there is an error in copying the data.
 *
 * @return Whether the data was copied into the input tensor successfully. Returns NO if an error
 *     occurred.
 */
- (BOOL)copyData:(NSData *)data error:(NSError **)error;

/**
 * Retrieves a copy of data in the tensor. For an output tensor, the data is only available after
 * the interpreter invocation has successfully completed; otherwise an error will be returned.
 *
 * @param error An optional error parameter populated when there is an error in retrieving the data.
 *
 * @return A copy of data in the tensor. `nil` if there is an error in retrieving the data or the
 *     data is not available.
 */
- (nullable NSData *)dataWithError:(NSError **)error;

/**
 * Retrieves the shape of the tensor, an array of positive unsigned integers containing the size
 * of each dimension. For example: the shape of [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]] is
 * [2, 2, 3] (i.e. an array of 2 arrays of 2 arrays of 3 numbers).
 *
 * @param error An optional error parameter populated when there is an error in retrieving the
 *     shape.
 *
 * @return The shape of the tensor. `nil` if there is an error in retrieving the shape.
 */
- (nullable NSArray<NSNumber *> *)shapeWithError:(NSError **)error;

@end

NS_ASSUME_NONNULL_END
