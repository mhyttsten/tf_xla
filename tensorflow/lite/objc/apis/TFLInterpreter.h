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
class MHTracer_DTPStensorflowPSlitePSobjcPSapisPSTFLInterpreterDTh {
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
   MHTracer_DTPStensorflowPSlitePSobjcPSapisPSTFLInterpreterDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSobjcPSapisPSTFLInterpreterDTh() {
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

@class TFLDelegate;
@class TFLInterpreterOptions;
@class TFLTensor;

NS_ASSUME_NONNULL_BEGIN

/**
 * @enum TFLInterpreterErrorCode
 * This enum specifies various error codes related to `TFLInterpreter`.
 */
typedef NS_ENUM(NSUInteger, TFLInterpreterErrorCode) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSobjcPSapisPSTFLInterpreterDTh mht_0(mht_0_v, 196, "", "./tensorflow/lite/objc/apis/TFLInterpreter.h", "NS_ENUM");

  /** Provided tensor index is invalid. */
  TFLInterpreterErrorCodeInvalidTensorIndex,

  /** Input data has invalid byte size. */
  TFLInterpreterErrorCodeInvalidInputByteSize,

  /** Provided shape is invalid. It must be a non-empty array of positive unsigned integers. */
  TFLInterpreterErrorCodeInvalidShape,

  /** Provided model cannot be loaded. */
  TFLInterpreterErrorCodeFailedToLoadModel,

  /** Failed to create `TFLInterpreter`. */
  TFLInterpreterErrorCodeFailedToCreateInterpreter,

  /** Failed to invoke `TFLInterpreter`. */
  TFLInterpreterErrorCodeFailedToInvoke,

  /** Failed to retrieve a tensor. */
  TFLInterpreterErrorCodeFailedToGetTensor,

  /** Invalid tensor. */
  TFLInterpreterErrorCodeInvalidTensor,

  /** Failed to resize an input tensor. */
  TFLInterpreterErrorCodeFailedToResizeInputTensor,

  /** Failed to copy data into an input tensor. */
  TFLInterpreterErrorCodeFailedToCopyDataToInputTensor,

  /** Copying data into an output tensor not allowed. */
  TFLInterpreterErrorCodeCopyDataToOutputTensorNotAllowed,

  /** Failed to get data from a tensor. */
  TFLInterpreterErrorCodeFailedToGetDataFromTensor,

  /** Failed to allocate memory for tensors. */
  TFLInterpreterErrorCodeFailedToAllocateTensors,

  /** Operation not allowed without allocating memory for tensors first. */
  TFLInterpreterErrorCodeAllocateTensorsRequired,

  /** Operation not allowed without invoking the interpreter first. */
  TFLInterpreterErrorCodeInvokeInterpreterRequired,
};

/**
 * A TensorFlow Lite model interpreter.
 *
 * Note: Interpreter instances are *not* thread-safe.
 */
@interface TFLInterpreter : NSObject

/** The total number of input tensors. 0 if the interpreter creation failed. */
@property(nonatomic, readonly) NSUInteger inputTensorCount;

/** The total number of output tensors. 0 if the interpreter creation failed. */
@property(nonatomic, readonly) NSUInteger outputTensorCount;

/** Unavailable. */
- (instancetype)init NS_UNAVAILABLE;

/**
 * Initializes a new TensorFlow Lite interpreter instance with the given model file path and the
 * default interpreter options.
 *
 * @param modelPath An absolute path to a TensorFlow Lite model file stored locally on the device.
 * @param error An optional error parameter populated when there is an error in initializing the
 *     interpreter.
 *
 * @return A new instance of `TFLInterpreter` with the given model and the default interpreter
 *     options. `nil` if there is an error in initializing the interpreter.
 */
- (nullable instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error;

/**
 * Initializes a new TensorFlow Lite interpreter instance with the given model file path and
 * options.
 *
 * @param modelPath An absolute path to a TensorFlow Lite model file stored locally on the device.
 * @param options Options to use for configuring the TensorFlow Lite interpreter.
 * @param error An optional error parameter populated when there is an error in initializing the
 *     interpreter.
 *
 * @return A new instance of `TFLInterpreter` with the given model and options. `nil` if there is an
 *     error in initializing the interpreter.
 */
- (nullable instancetype)initWithModelPath:(NSString *)modelPath
                                   options:(TFLInterpreterOptions *)options
                                     error:(NSError **)error;

/**
 * Initializes a new TensorFlow Lite interpreter instance with the given model file path, options
 * and delegates.
 *
 * @param modelPath An absolute path to a TensorFlow Lite model file stored locally on the device.
 * @param options Options to use for configuring the TensorFlow Lite interpreter.
 * @param delegates Delegates to use with the TensorFlow Lite interpreter. When the array is empty,
 *     no delegate will be applied.
 * @param error An optional error parameter populated when there is an error in initializing the
 *     interpreter.
 *
 * @return A new instance of `TFLInterpreter` with the given model and options. `nil` if there is an
 *     error in initializing the interpreter.
 */
- (nullable instancetype)initWithModelPath:(NSString *)modelPath
                                   options:(TFLInterpreterOptions *)options
                                 delegates:(NSArray<TFLDelegate *> *)delegates
                                     error:(NSError **)error NS_DESIGNATED_INITIALIZER;

/**
 * Invokes the interpreter to run inference.
 *
 * @param error An optional error parameter populated when there is an error in invoking the
 *     interpreter.
 *
 * @return Whether the invocation is successful. Returns NO if an error occurred.
 */
- (BOOL)invokeWithError:(NSError **)error;

/**
 * Returns the input tensor at the given index.
 *
 * @param index The index of an input tensor.
 * @param error An optional error parameter populated when there is an error in looking up the input
 *     tensor.
 *
 * @return The input tensor at the given index. `nil` if there is an error. See the `TFLTensor`
 *     class documentation for more details on the life expectancy between the returned tensor and
 *     this interpreter.
 */
- (nullable TFLTensor *)inputTensorAtIndex:(NSUInteger)index error:(NSError **)error;

/**
 * Returns the output tensor at the given index.
 *
 * @param index The index of an output tensor.
 * @param error An optional error parameter populated when there is an error in looking up the
 *     output tensor.
 *
 * @return The output tensor at the given index. `nil` if there is an error. See the `TFLTensor`
 *     class documentation for more details on the life expectancy between the returned tensor and
 *     this interpreter.
 */
- (nullable TFLTensor *)outputTensorAtIndex:(NSUInteger)index error:(NSError **)error;

/**
 * Resizes the input tensor at the given index to the specified shape (an array of positive unsigned
 * integers).
 *
 * @param index The index of an input tensor.
 * @param shape Shape that the given input tensor should be resized to. It should be an array of
 *     positive unsigned integer(s) containing the size of each dimension.
 * @param error An optional error parameter populated when there is an error in resizing the input
 *     tensor.
 *
 * @return Whether the input tensor was resized successfully. Returns NO if an error occurred.
 */
- (BOOL)resizeInputTensorAtIndex:(NSUInteger)index
                         toShape:(NSArray<NSNumber *> *)shape
                           error:(NSError **)error;

/**
 * Allocates memory for tensors.
 *
 * @param error An optional error parameter populated when there is an error in allocating memory.
 *
 * @return Whether memory allocation is successful. Returns NO if an error occurred.
 */
- (BOOL)allocateTensorsWithError:(NSError **)error;

@end

NS_ASSUME_NONNULL_END
