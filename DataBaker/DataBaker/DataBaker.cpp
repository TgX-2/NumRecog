// DataBaker.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include <filesystem>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/opencv.hpp" 

using namespace cv;
using namespace std;
using namespace filesystem;

Size exlarge(1000, 1000);
Size large(500, 500);
Size med(200, 200);
Size small(100, 100);
Size exsmall(50, 50);
Size standard(28, 28);

string input;
string output;

void proc(Mat& src, string name)
{
	int ch = src.channels();

	int mnx = src.rows, mny = src.cols, mxx = 0, mxy = 0;
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
		{
			if (src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i, j)[2] < 600)
			{
				src.at<Vec3b>(i, j)[0] = 0;
				src.at<Vec3b>(i, j)[1] = 0;
				src.at<Vec3b>(i, j)[2] = 0;

				mnx = min(mnx, j);
				mny = min(mny, i);

				mxx = max(mxx, j);
				mxy = max(mxy, i);
			}
			else
			{
				src.at<Vec3b>(i, j)[0] = 255;
				src.at<Vec3b>(i, j)[1] = 255;
				src.at<Vec3b>(i, j)[2] = 255;
			}
		}

	Mat out = src(Range(mny, mxy + 1), Range(mnx, mxx + 1));

	resize(out, out, exlarge);

	for (int i = 0; i < out.rows; i++)
		for (int j = 0; j < out.cols; j++)
		{
			if (out.at<Vec3b>(i, j)[0] + out.at<Vec3b>(i, j)[1] + out.at<Vec3b>(i, j)[2] < 600)
			{
				out.at<Vec3b>(i, j)[0] = 0;
				out.at<Vec3b>(i, j)[1] = 0;
				out.at<Vec3b>(i, j)[2] = 0;
			}
			else
			{
				out.at<Vec3b>(i, j)[0] = 255;
				out.at<Vec3b>(i, j)[1] = 255;
				out.at<Vec3b>(i, j)[2] = 255;
			}
		}

	resize(out, out, large);

	for (int i = 0; i < out.rows; i++)
		for (int j = 0; j < out.cols; j++)
		{
			if (out.at<Vec3b>(i, j)[0] + out.at<Vec3b>(i, j)[1] + out.at<Vec3b>(i, j)[2] < 600)
			{
				out.at<Vec3b>(i, j)[0] = 0;
				out.at<Vec3b>(i, j)[1] = 0;
				out.at<Vec3b>(i, j)[2] = 0;
			}
			else
			{
				out.at<Vec3b>(i, j)[0] = 255;
				out.at<Vec3b>(i, j)[1] = 255;
				out.at<Vec3b>(i, j)[2] = 255;
			}
		}

	resize(out, out, med);

	for (int i = 0; i < out.rows; i++)
		for (int j = 0; j < out.cols; j++)
		{
			if (out.at<Vec3b>(i, j)[0] + out.at<Vec3b>(i, j)[1] + out.at<Vec3b>(i, j)[2] < 600)
			{
				out.at<Vec3b>(i, j)[0] = 0;
				out.at<Vec3b>(i, j)[1] = 0;
				out.at<Vec3b>(i, j)[2] = 0;
			}
			else
			{
				out.at<Vec3b>(i, j)[0] = 255;
				out.at<Vec3b>(i, j)[1] = 255;
				out.at<Vec3b>(i, j)[2] = 255;
			}
		}

	resize(out, out, small);

	for (int i = 0; i < out.rows; i++)
		for (int j = 0; j < out.cols; j++)
		{
			if (out.at<Vec3b>(i, j)[0] + out.at<Vec3b>(i, j)[1] + out.at<Vec3b>(i, j)[2] < 600)
			{
				out.at<Vec3b>(i, j)[0] = 0;
				out.at<Vec3b>(i, j)[1] = 0;
				out.at<Vec3b>(i, j)[2] = 0;
			}
			else
			{
				out.at<Vec3b>(i, j)[0] = 255;
				out.at<Vec3b>(i, j)[1] = 255;
				out.at<Vec3b>(i, j)[2] = 255;
			}
		}

	resize(out, out, exsmall);

	for (int i = 0; i < out.rows; i++)
		for (int j = 0; j < out.cols; j++)
		{
			if (out.at<Vec3b>(i, j)[0] + out.at<Vec3b>(i, j)[1] + out.at<Vec3b>(i, j)[2] < 600)
			{
				out.at<Vec3b>(i, j)[0] = 0;
				out.at<Vec3b>(i, j)[1] = 0;
				out.at<Vec3b>(i, j)[2] = 0;
			}
			else
			{
				out.at<Vec3b>(i, j)[0] = 255;
				out.at<Vec3b>(i, j)[1] = 255;
				out.at<Vec3b>(i, j)[2] = 255;
			}
		}

	resize(out, out, standard);

	for (int i = 0; i < out.rows; i++)
		for (int j = 0; j < out.cols; j++)
		{
			if (out.at<Vec3b>(i, j)[0] + out.at<Vec3b>(i, j)[1] + out.at<Vec3b>(i, j)[2] < 600)
			{
				out.at<Vec3b>(i, j)[0] = 0;
				out.at<Vec3b>(i, j)[1] = 0;
				out.at<Vec3b>(i, j)[2] = 0;
			}
			else
			{
				out.at<Vec3b>(i, j)[0] = 255;
				out.at<Vec3b>(i, j)[1] = 255;
				out.at<Vec3b>(i, j)[2] = 255;
			}
		}

	imwrite(name, out);
	std::cout << "Saved to " << name << endl;
}

void rmopa(Mat& src)
{
	if (src.channels() == 4)
	{
		Mat bg(src.size(), CV_8UC3, Scalar(255, 255, 255));
		Mat rgb, alpha;

		vector<Mat> ch;
		split(src, ch); // BGRA
		alpha = ch[3];
		merge(vector<Mat>{ch[0], ch[1], ch[2]}, rgb);

		rgb.copyTo(bg, alpha); // chỉ copy pixel alpha > 0
		src = bg;
	}
}

string itos(int i)
{
	if (i == 0) return "0";
	string s = "";
	while (i)
	{
		s = char(i % 10 + '0') + s;
		i /= 10;
	}
	return s;
}

int main()
{
	path ou = current_path() / ".." / ".." / "imgset" / "baked";
	ou = weakly_canonical(ou); ou += "/";
	path in = current_path() / ".." / ".." / "imgset" / "rawdata";
	in = weakly_canonical(in); in += "/";

	string input = in.string().c_str();
	string output = ou.string().c_str();

	for (int i = 5; i < 10; i++)
	{
		string ii = input + itos(i) + "/" + itos(i) + "/";
		string oo = output + itos(i) + "/";

		int cnt = 0;
		for (const auto& xx : directory_iterator(oo)) ++cnt;

		for (const auto& entry : directory_iterator(ii))
		{
			std::cout << "Processing " << entry.path().string() << endl;

			string filepath = entry.path().string();
			Mat src = imread(filepath, IMREAD_UNCHANGED);

			rmopa(src); ++cnt; if (cnt > 200) break;
			proc(src, oo + itos(cnt) + ".jpg");
			remove(filepath);
		}
	}

	return 0;
}