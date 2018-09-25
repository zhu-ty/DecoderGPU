#ifdef _WIN32
#include <io.h> 
#define access    _access_s
#else
#include <unistd.h>
#endif

// include std
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <queue>
#include <thread>
#include <memory>

// opencv
#include <opencv2/opencv.hpp>

// cuda
#ifdef _WIN32
#include <windows.h>
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include <dynlink_nvcuvid.h>
#include <dynlink_cuviddec.h>
#include <device_launch_parameters.h>

#include "NvEncoder/NvEncoderCuda.h"
#include "Utils/Logger.h"
#include "Utils/NvEncoderCLIOptions.h"
#include "Utils/NvCodecUtils.h"

#include "NPPJpegCoder.h"

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

inline bool isFileExists(const std::string &Filename) {
	return access(Filename.c_str(), 0) == 0;
}

int main(int argc, char* argv[]) 
{
	if (argc < 2)
	{
		printf("usage: Decoder input.bin gpu_id\n");
		return 0;
	}
	int iGpu = 0;
	if (argc >= 3)
	{
		iGpu = std::atoi(argv[2]);
	}

	//Encoder init
	//TODO: maybe use YUV420 in the future
	NV_ENC_BUFFER_FORMAT eFormat = NV_ENC_BUFFER_FORMAT_IYUV;
	//NV_ENC_BUFFER_FORMAT eFormat = NV_ENC_BUFFER_FORMAT_ABGR;
	NvEncoderInitParam encodeCLIOptions;
	//NvEncGetEncodeCaps();
	ck(cuInit(0));
	CUdevice cuDevice = 0;
	int nGpu = 0;
	ck(cuDeviceGetCount(&nGpu));
	if (iGpu >= nGpu)
	{
		iGpu = nGpu - 1;
	}


	ck(cuDeviceGet(&cuDevice, iGpu));
	char szDeviceName[80];
	ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
	std::cout << "GPU in use: " << szDeviceName << std::endl;
	CUcontext cuContext = NULL;
	ck(cuCtxCreate(&cuContext, 0, cuDevice));


	std::string videoname(argv[1]);
	int dot_pos = videoname.find_last_of('.');
	std::string output = videoname.substr(0, dot_pos) + ".h264";
	std::cout << cv::format("Find bin file: %s\n", videoname.c_str()) << std::endl;

	int frameNum;
	int quality;
	int width;
	int height;
	FILE* fp = fopen(videoname.c_str(), "rb");
	fread(&frameNum, sizeof(unsigned int), 1, fp);

	if (argc >= 4)
	{
		frameNum = std::atoi(argv[3]);
	}


	fread(&width, sizeof(int), 1, fp);
	fread(&height, sizeof(int), 1, fp);
	fread(&quality, sizeof(int), 1, fp);


	npp::NPPJpegCoder coder;
	unsigned int length;
	char* data = new char[width * height];
	cv::cuda::GpuMat img(height, width, CV_8UC3);
	cv::cuda::GpuMat img_4(height, width, CV_8UC4);
	std::vector<void*> _gpu_decoded_YUVdata(3);
	int Ystep;



	std::ofstream fpOut(output.c_str(), std::ios::out | std::ios::binary);
	NvEncoderCuda enc(cuContext, width, height, eFormat);
	NV_ENC_INITIALIZE_PARAMS initializeParams = { NV_ENC_INITIALIZE_PARAMS_VER };
	NV_ENC_CONFIG encodeConfig = { NV_ENC_CONFIG_VER };
	initializeParams.encodeConfig = &encodeConfig;
	enc.CreateDefaultEncoderParams(&initializeParams, encodeCLIOptions.GetEncodeGUID(), encodeCLIOptions.GetPresetGUID());
	encodeCLIOptions.SetInitParams(&initializeParams, eFormat);
	enc.CreateEncoder(&initializeParams);
	int nFrameSize = enc.GetFrameSize();
	//std::unique_ptr<uint8_t[]> pHostFrame(new uint8_t[nFrameSize]);
	int nFrame = 0;
	
	
	
	
	//uint32_t t;
	//NvEncGetEncodePresetCount((void *)&enc, encodeCLIOptions.GetEncodeGUID(), &t);




	for (size_t i = 0; i < frameNum; i++) {
		//printf("Decode frame %d, total %d frames.\n", i, frameNum);
		if (i % 50 == 0)
			printf("%s : Frame:%d, Total:%d\n", videoname.c_str(), i, frameNum);
		fread(&length, sizeof(unsigned int), 1, fp);
		fread(data, length, 1, fp);
		if (i == 0) {
			coder.init(width, height, quality);
		}
		//coder.decode(reinterpret_cast<unsigned char*>(data), length, img_4, 2);
		coder.decode(reinterpret_cast<unsigned char*>(data), length, _gpu_decoded_YUVdata, Ystep);

		// For receiving encoded packets
		std::vector<std::vector<uint8_t>> vPacket;
		if (i < frameNum - 1)
		{
			const NvEncInputFrame* encoderInputFrame = enc.GetNextInputFrame();
			//NvEncoderCuda::CopyToDeviceFrame(cuContext, img_4.data, img_4.step, (CUdeviceptr)encoderInputFrame->inputPtr,
			//	(int)encoderInputFrame->pitch,
			//	enc.GetEncodeWidth(),
			//	enc.GetEncodeHeight(),
			//	CU_MEMORYTYPE_DEVICE,
			//	//CU_MEMORYTYPE_HOST,
			//	encoderInputFrame->bufferFormat,
			//	encoderInputFrame->chromaOffsets,
			//	encoderInputFrame->numChromaPlanes);
			NvEncoderCuda::CopyToDeviceFrame_YUV420(cuContext, _gpu_decoded_YUVdata, Ystep, (CUdeviceptr)encoderInputFrame->inputPtr,
				(int)encoderInputFrame->pitch,
				enc.GetEncodeWidth(),
				enc.GetEncodeHeight(),
				CU_MEMORYTYPE_DEVICE,
				//CU_MEMORYTYPE_HOST,
				encoderInputFrame->bufferFormat,
				encoderInputFrame->chromaOffsets,
				encoderInputFrame->numChromaPlanes);

			enc.EncodeFrame(vPacket);
		}
		else
		{
			enc.EndEncode(vPacket);
		}

		nFrame += (int)vPacket.size();
		for (std::vector<uint8_t> &packet : vPacket)
		{
			// For each encoded packet
			fpOut.write(reinterpret_cast<char*>(packet.data()), packet.size());
		}
	}

	enc.DestroyEncoder();
	fpOut.close();

	fclose(fp);
	coder.release();
	printf("\n");

	return 0;
}