#include "data.h"


cv::Mat letterbox(cv::Mat image, cv::Size s, int borderThre, bool center )
{
	int imageH = image.rows, imageW = image.cols;
	cv::Mat output = cv::Mat(s, CV_8UC3);;

	float ratioW = s.width / (imageW * 1.0), ratioH = s.height / (imageH * 1.0);
	float r = std::min(ratioW, ratioH);
	int new_unpad_w = int(imageW * r);
	int new_unpad_h = int(imageH * r);

	int padH = s.height - new_unpad_h;
	int padW = s.width - new_unpad_w;

	// center 中心往外pad，对称
	if (center)
	{
		padW /= 2.0;
		padH /= 2.0;
	}

	//pad
	if (s.width != imageW || s.height != imageH)
	{
		cv::Mat temp;
		cv::resize(image, temp, cv::Size(new_unpad_w, new_unpad_h));

		int top = 0;
		int bottom = int(padH + 0.1);
		int left = 0;
		int right = int(padW + 0.1);

		if (center)
		{
			top = int(padH);
			bottom = int(padH);
			left = int(padW);
			right = int(padW);
		}
		cv::copyMakeBorder(temp, output, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(borderThre, borderThre, borderThre));
	}
	else
	{
		image.convertTo(output, output.type());
	}

	return output;
}



cv::Mat letterboxOut(cv::Mat mask, cv::Size imageSize, cv::Size inputSize, bool center)
{
	if (mask.empty())
		return mask;


	int ih = imageSize.height, iw = imageSize.width;
	int mh = inputSize.height, mw = inputSize.width;

	if (ih == mh && iw == mw)
	{
		return mask;
	}
	float r = std::min(mw * 1.0 / iw, mh * 1.0 / ih);
	int new_unpad_w = int(iw * r);
	int new_unpad_h = int(ih * r);

	int padH = mh - new_unpad_h;
	int padW = mw - new_unpad_w;

	// center 中心往外pad，对称
	if (center)
	{
		padW /= 2.0;
		padH /= 2.0;
	}

	int top = 0;
	int bottom = int(padH + 0.1);
	int left = 0;
	int right = int(padW + 0.1);

	if (center)
	{
		top = int(padH);
		bottom = int(padH);
		left = int(padW);
		right = int(padW);
	}

	//cv::Mat cropMask = cv::Mat::zeros(m, CV_8UC1);
	cv::Rect cropRect = cv::Rect(cv::Point(left, top), cv::Point(mw - right, mh - bottom));

	cv::Mat out = cv::Mat::zeros(imageSize, mask.type());
	cv::Mat temp_out = mask;
	cv::resize(temp_out(cropRect), out, imageSize);
	return out;

}

