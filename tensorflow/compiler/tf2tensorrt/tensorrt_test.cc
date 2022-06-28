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
class MHTracer_DTPStensorflowPScompilerPStf2tensorrtPStensorrt_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPStensorrt_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2tensorrtPStensorrt_testDTcc() {
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
#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include <functional>
#include <numeric>
#include <stack>

#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "tensorflow/compiler/tf2tensorrt/common/utils.h"
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_logger.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/test.h"
#include "third_party/tensorrt/NvInfer.h"
#include "third_party/tensorrt/NvInferPlugin.h"
#include "third_party/tensorrt/NvInferRuntimeCommon.h"

#ifdef TF_TRT_USE_EFFICIENT_NMS_PLUGIN
#include "third_party/tensorrt/plugin/efficientNMSPlugin/efficientNMSPlugin.h"
namespace tensorflow {
namespace tensorrt {
std::unique_ptr<nvinfer1::plugin::EfficientNMSPluginCreator>
MakeNMSPluginCreator(const std::string& plugin_namespace = "tftrt") {
  auto pluginCreator =
      std::make_unique<nvinfer1::plugin::EfficientNMSPluginCreator>();
  pluginCreator->setPluginNamespace(plugin_namespace.c_str());
  std::string pluginType = std::string{pluginCreator->getPluginNamespace()} +
                           "::" + std::string{pluginCreator->getPluginName()} +
                           " version " +
                           std::string{pluginCreator->getPluginVersion()};
  VLOG(0) << "Created plugin type " << pluginType;
  return pluginCreator;
}

struct PluginDeleter {
  void operator()(nvinfer1::IPluginV2* t);
};

void PluginDeleter::operator()(nvinfer1::IPluginV2* t) { t->destroy(); }

std::unique_ptr<nvinfer1::IPluginV2, PluginDeleter> createPlugin(
    const std::string& name, nvinfer1::IPluginCreator* pluginCreator,
    const std::vector<nvinfer1::PluginField>& pluginFields) {
  if (!pluginCreator) {
    return nullptr;
  }
  nvinfer1::PluginFieldCollection fc;
  fc.nbFields = pluginFields.size();
  fc.fields = pluginFields.data();
  return std::unique_ptr<nvinfer1::IPluginV2, PluginDeleter>{
      pluginCreator->createPlugin(name.c_str(), &fc)};
}
}  // namespace tensorrt
}  // namespace tensorflow
#endif

namespace tensorflow {
namespace tensorrt {

class ScopedWeights {
 public:
  ScopedWeights(float value) : value_(value) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPStensorrt_testDTcc mht_0(mht_0_v, 246, "", "./tensorflow/compiler/tf2tensorrt/tensorrt_test.cc", "ScopedWeights");

    w.type = nvinfer1::DataType::kFLOAT;
    w.values = &value_;
    w.count = 1;
  }
  const nvinfer1::Weights& get() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPStensorrt_testDTcc mht_1(mht_1_v, 254, "", "./tensorflow/compiler/tf2tensorrt/tensorrt_test.cc", "get");
 return w; }

 private:
  float value_;
  nvinfer1::Weights w;
};

class ScopedShapedWeights {
 public:
  ScopedShapedWeights(nvinfer1::Dims dims, float value)
      : dims_(dims),
        value_(std::accumulate(dims.d, dims.d + dims.nbDims, 1,
                               std::multiplies<>()),
               value) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPStensorrt_testDTcc mht_2(mht_2_v, 270, "", "./tensorflow/compiler/tf2tensorrt/tensorrt_test.cc", "ScopedShapedWeights");

    w.type = nvinfer1::DataType::kFLOAT;
    w.values = value_.data();
    w.count = value_.size();
  }

  nvinfer1::Dims dims_;
  std::vector<float> value_;
  nvinfer1::Weights w;
};

const char* kInputTensor1 = "input1";
const char* kInputTensor2 = "input2";
const char* kOutputTensor1 = "output";
const char* kOutputTensor2 = "output-nms";

// Creates a network to compute x+y.
TrtUniquePtrType<nvinfer1::IHostMemory> CreateSerializedEngine() {
  Logger& logger = *Logger::GetLogger();
  TrtUniquePtrType<nvinfer1::IBuilder> builder(
      nvinfer1::createInferBuilder(logger));
  TrtUniquePtrType<nvinfer1::INetworkDefinition> network(
      builder->createNetworkV2(
          1U << static_cast<uint32_t>(
              nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
  // Add the input.
  auto input1 = network->addInput(kInputTensor1, nvinfer1::DataType::kFLOAT,
                                  nvinfer1::Dims4{1, 1, 1, 1});
  auto input2 = network->addInput(kInputTensor2, nvinfer1::DataType::kFLOAT,
                                  nvinfer1::Dims4{1, 1, 1, 1});
  EXPECT_NE(input1, nullptr);
  EXPECT_NE(input2, nullptr);
  // Add an ILayer layer.
  auto layer = network->addElementWise(*input1, *input2,
                                       nvinfer1::ElementWiseOperation::kSUM);
  EXPECT_NE(layer, nullptr);
  auto output = layer->getOutput(0);
  output->setName(kOutputTensor1);
  network->markOutput(*output);

#ifdef TF_TRT_USE_EFFICIENT_NMS_PLUGIN
  // Add an efficient nms plugin.
  ScopedShapedWeights boxes_weights(nvinfer1::Dims3(1, 10, 4), 0.0f);
  ScopedShapedWeights scores_weights(nvinfer1::Dims3(1, 10, 10), 0.0f);
  nvinfer1::IConstantLayer* boxes =
      network->addConstant(boxes_weights.dims_, boxes_weights.w);
  nvinfer1::IConstantLayer* scores =
      network->addConstant(scores_weights.dims_, scores_weights.w);

  std::array<nvinfer1::ITensor*, 2> nms_inputs = {boxes->getOutput(0),
                                                  scores->getOutput(0)};
  auto plugin_creator = MakeNMSPluginCreator("tftrt");
  auto plugin = createPlugin("nms_plugin_instance", plugin_creator.get(), {});
  auto nms = network->addPluginV2(nms_inputs.data(), 2, *plugin);
  nms->getOutput(0)->setName(kOutputTensor2);
  network->markOutput(*nms->getOutput(0));
#else
  auto sub_layer = network->addElementWise(
      *input1, *input2, nvinfer1::ElementWiseOperation::kSUB);
  EXPECT_NE(sub_layer, nullptr);
  network->markOutput(*sub_layer->getOutput(0));
  sub_layer->getOutput(0)->setName(kOutputTensor2);
#endif

  // Build the engine.
  builder->setMaxBatchSize(1);
  TrtUniquePtrType<nvinfer1::IBuilderConfig> builderConfig(
      builder->createBuilderConfig());
  builderConfig->setMaxWorkspaceSize(1 << 20);
  TrtUniquePtrType<nvinfer1::ICudaEngine> engine(
      builder->buildEngineWithConfig(*network, *builderConfig));
  EXPECT_NE(engine, nullptr);
  // Serialize the engine to create a model, then close everything.
  TrtUniquePtrType<nvinfer1::IHostMemory> model(engine->serialize());
  return model;
}

template <typename T>
unsigned GetBindingSizeBytes(const nvinfer1::ICudaEngine& engine, int index,
                             unsigned batch_size) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPStensorrt_testDTcc mht_3(mht_3_v, 352, "", "./tensorflow/compiler/tf2tensorrt/tensorrt_test.cc", "GetBindingSizeBytes");

  unsigned vol = batch_size;
  auto dims = engine.getBindingDimensions(index);
  int vecDim = engine.getBindingVectorizedDim(index);
  if (-1 != vecDim)  // i.e., 0 != lgScalarsPerVector
  {
    int scalarsPerVec = engine.getBindingComponentsPerElement(index);
    // Divide round up.
    dims.d[vecDim] = (dims.d[vecDim] + scalarsPerVec - 1 / scalarsPerVec);
    vol *= scalarsPerVec;
  }
  vol *= std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<>());
  return vol * sizeof(T);
}

// Executes the network.
void Execute(nvinfer1::IExecutionContext* context, const float* input1,
             const float* input2, float* output1, float* output2) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPStensorrt_testDTcc mht_4(mht_4_v, 372, "", "./tensorflow/compiler/tf2tensorrt/tensorrt_test.cc", "Execute");

  const nvinfer1::ICudaEngine& engine = context->getEngine();

  // We have two bindings: input and output.
  ASSERT_EQ(engine.getNbBindings(), 4);
  const int input_index1 = engine.getBindingIndex(kInputTensor1);
  const int input_index2 = engine.getBindingIndex(kInputTensor2);
  const int output_index1 = engine.getBindingIndex(kOutputTensor1);
  const int output_index2 = engine.getBindingIndex(kOutputTensor2);

  // Create GPU buffers and a stream
  std::vector<void*> buffers(engine.getNbBindings());
  for (int i = 0; i < buffers.size(); i++) {
    ASSERT_EQ(
        0, cudaMalloc(&buffers[i], GetBindingSizeBytes<float>(engine, i, 1)));
  }

  cudaStream_t stream;
  ASSERT_EQ(0, cudaStreamCreate(&stream));

  // Copy the input to the GPU, execute the network, and copy the output back.
  //
  // Note that since the host buffer was not created as pinned memory, these
  // async copies are turned into sync copies. So the following synchronization
  // could be removed.
  ASSERT_EQ(0, cudaMemcpyAsync(buffers[input_index1], input1, sizeof(float),
                               cudaMemcpyHostToDevice, stream));
  ASSERT_EQ(0, cudaMemcpyAsync(buffers[input_index2], input2, sizeof(float),
                               cudaMemcpyHostToDevice, stream));
  context->enqueueV2(buffers.data(), stream, nullptr);
  ASSERT_EQ(0, cudaMemcpyAsync(output1, buffers[output_index1], sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
  ASSERT_EQ(
      0, cudaMemcpyAsync(output2, buffers[output_index2],
                         GetBindingSizeBytes<int32>(engine, output_index2, 1),
                         cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);

  // Release the stream and the buffers
  for (int i = 0; i < buffers.size(); i++) {
    ASSERT_EQ(0, cudaFree(buffers[i]));
  }
  cudaStreamDestroy(stream);
}

TEST(TensorrtTest, BasicFunctions) {
  // We must register the plugin creator in order to deserialize the plugin.
#ifdef TF_TRT_USE_EFFICIENT_NMS_PLUGIN
  auto plugin_creator = MakeNMSPluginCreator("tftrt");
  getPluginRegistry()->registerCreator(*plugin_creator, "tftrt");
#endif

  // Handle the case where the test is run on machine with no gpu available.
  if (CHECK_NOTNULL(GPUMachineManager())->VisibleDeviceCount() <= 0) {
    LOG(WARNING) << "No gpu device available, probably not being run on a gpu "
                    "machine. Skipping...";
    return;
  }

  // Create a serialized engine
  TrtUniquePtrType<nvinfer1::IHostMemory> model = CreateSerializedEngine();
  // Use the model to create an engine and then an execution context.
  Logger& logger = *Logger::GetLogger();
  TrtUniquePtrType<nvinfer1::IRuntime> runtime(
      nvinfer1::createInferRuntime(logger));
  TrtUniquePtrType<nvinfer1::ICudaEngine> engine(
      runtime->deserializeCudaEngine(model->data(), model->size(), nullptr));
  TrtUniquePtrType<nvinfer1::IExecutionContext> context(
      engine->createExecutionContext());

  // Execute the network.
  float input1 = 1234;
  float input2 = 567;

  std::vector<float> output1(
      GetBindingSizeBytes<float>(*engine, 2, 1) / sizeof(float), 0.0f);

  std::vector<float> output2(
      GetBindingSizeBytes<int32>(*engine, 3, 1) / sizeof(int32), 0.0f);

  ASSERT_EQ(output1.size(), 1);
  ASSERT_EQ(output2.size(), 1);

  Execute(context.get(), &input1, &input2, output1.data(), output2.data());
  EXPECT_EQ(output1[0], input1 + input2);

#ifdef TF_TRT_USE_EFFICIENT_NMS_PLUGIN
  EXPECT_EQ(output2[0], 0);
#else
  EXPECT_EQ(output2[0], 667);
#endif  // TF_TRT_USE_EFFICIENT_NMS_PLUGIN
}

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
