// ImgReader.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <filesystem>

using namespace std;
using namespace cv;
using namespace filesystem;

vector<int> arr;

string itos(int n)
{
	if (n == 0) return "0";
	string res = "";
	while (n)
	{
		res = char(n % 10 + '0') + res;
		n /= 10;
	}
	return res;
}

void proc()
{
	path ou = current_path() / ".." / ".." / "imgset" / "number_input" / "inp.inp";
	ou = weakly_canonical(ou);
	path in = current_path() / ".." / ".." / "imgset" / "baked";
	in = weakly_canonical(in); in += "/";

	ofstream out(ou.string().c_str());
	string inp(in.string().c_str());
	
	int cnt = 0;
	for (int i = 0; i < 10; i++)
	{
		int d = 0;
		for (const auto& entry : directory_iterator(inp + itos(i) + '/'))
		{
			++cnt;
			++d;
			if (d == 1000) break;
		}
		cout << d << endl;
	}

	out << cnt<<endl;

	for (int i = 0; i < 10; i++)
	{
		int d = 0;
		for (const auto& entry : directory_iterator(inp + itos(i) + '/'))
		{
			++d;
			Mat img = imread(entry.path().string(), IMREAD_GRAYSCALE);

			for (int i = 0; i < img.cols; i++)
				for (int j = 0; j < img.rows; j++)
				{
					uchar pixel_value = img.at<uchar>(i, j);
					if (pixel_value > 128)
						out << "0 ";
					else
						out << "1 ";
				}
			out << i<<' '<<'\n';
			if (d == 1000)break;
		}
	}
}

int main()
{
	proc();
	return 0;
}
