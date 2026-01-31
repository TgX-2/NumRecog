// Kmean.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <map>
#include <algorithm>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <filesystem>

#define pb push_back
#define ll long long
#define pii pair<int, int>
#define fi first
#define se second

using namespace std;
using namespace filesystem;

const int vec = 25;
const int cennum = 200;
const double pp[10] = { 1,2,1,0.7,1,3,1,1.8,1.8,1.2};

cv::Mat mean(1, 784, CV_64F), eigenval(vec, 1, CV_64F), eigenvector(vec, 784, CV_64F), cen(cennum, vec, CV_64F), centag(cennum, 1, CV_64F);
vector<long double> mt;
vector<vector<long double>> arr;
cv::PCA pca;

cv::Size exlarge(1000, 1000);
cv::Size large(500, 500);
cv::Size med(200, 200);
cv::Size small(100, 100);
cv::Size exsmall(50, 50);
cv::Size standard(28, 28);

double ss[10];

bool o = 1;

void preproc()
{
	path ii = current_path() / ".." / ".." / "imgset" / "cluster.txt";
	ii = weakly_canonical(ii);

	ifstream in(ii.string().c_str());
    
    string st;
    while (in >> st)
    {
        string s = "", z = ""; bool o = 0;
        for (int i = 0; i < st.length(); i++)
        {
            if ((st[i] >= '0' && st[i] <= '9') || st[i] == '.' || st[i] == '-' || (o&&st[i]=='e')) s += st[i];
            if (st[i]=='[') z += st[i];

            if (st[i] >= '0' && st[i] <= '9') o = 1;
        }

        if (z != "") arr.pb(mt);

		long double val = std::stold(s);;
        arr[arr.size() - 1].pb(val);
    }

    for (int i = 0; i < 784; i++) mean.at<double>(0, i) = arr[0][i];
    for (int i = 0; i < vec; i++)
    for (int j = 0; j < 784; j++) eigenvector.at<double>(i, j) = arr[1][i * 784 + j];
    for (int i = 0; i < vec; i++) eigenval.at<double>(i, 0) = arr[2][i];
    for (int i = 0; i < cennum; i++)
    for (int j = 0; j < vec; j++) cen.at<double>(i, j) = arr[3][i * vec + j];
    for (int i = 0; i < cennum; i++) centag.at<double>(i, 0) = arr[4][i];


	cout << arr[0].size() << endl;
	cout << arr[1].size() << endl;
	cout << arr[2].size() << endl;
	cout << arr[3].size() << endl;
	cout << arr[4].size() << endl;
	cout << cennum << ' ' << vec << endl;
	//cin.get();

	pca.eigenvalues = eigenval;
	pca.eigenvectors = eigenvector;
	pca.mean = mean;
}

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

void read(string path)
{
    ofstream out("inp.inp");

	if (path == "") for (const auto& entry : directory_iterator("inp"))
	{
		cout << entry.path().string();
		path = entry.path().string();
	}

	cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);

	for (int i = 0; i < img.cols; i++)
		for (int j = 0; j < img.rows; j++)
		{
			uchar pixel_value = img.at<uchar>(i, j);
		if (pixel_value > 128)
			out << "0 ";
		else
			out << "1 ";
	}
}

void bake(cv::Mat& src)
{
	int ch = src.channels();

	int mnx = src.rows, mny = src.cols, mxx = 0, mxy = 0;
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
		{
			if (src.at<cv::Vec3b>(i, j)[0] + src.at<cv::Vec3b>(i, j)[1] + src.at<cv::Vec3b>(i, j)[2] < 600)
			{
				src.at<cv::Vec3b>(i, j)[0] = 0;
				src.at<cv::Vec3b>(i, j)[1] = 0;
				src.at<cv::Vec3b>(i, j)[2] = 0;

				mnx = min(mnx, j);
				mny = min(mny, i);

				mxx = max(mxx, j);
				mxy = max(mxy, i);
			}
			else
			{
				src.at<cv::Vec3b>(i, j)[0] = 255;
				src.at<cv::Vec3b>(i, j)[1] = 255;
				src.at<cv::Vec3b>(i, j)[2] = 255;
			}
		}

	if(mnx==src.rows) 	{
		mnx = 0; mny = 0; mxx = src.cols - 1; mxy = src.rows - 1;
	}
	cv::Mat out = src(cv::Range(mny, mxy+1), cv::Range(mnx, mxx+1));

	resize(out, out, exlarge);

	for (int i = 0; i < out.rows; i++)
		for (int j = 0; j < out.cols; j++)
		{
			if (out.at<cv::Vec3b>(i, j)[0] + out.at<cv::Vec3b>(i, j)[1] + out.at<cv::Vec3b>(i, j)[2] < 600)
			{
				out.at<cv::Vec3b>(i, j)[0] = 0;
				out.at<cv::Vec3b>(i, j)[1] = 0;
				out.at<cv::Vec3b>(i, j)[2] = 0;
			}
			else
			{
				out.at<cv::Vec3b>(i, j)[0] = 255;
				out.at<cv::Vec3b>(i, j)[1] = 255;
				out.at<cv::Vec3b>(i, j)[2] = 255;
			}
		}

	resize(out, out, large);

	for (int i = 0; i < out.rows; i++)
		for (int j = 0; j < out.cols; j++)
		{
			if (out.at<cv::Vec3b>(i, j)[0] + out.at<cv::Vec3b>(i, j)[1] + out.at<cv::Vec3b>(i, j)[2] < 600)
			{
				out.at<cv::Vec3b>(i, j)[0] = 0;
				out.at<cv::Vec3b>(i, j)[1] = 0;
				out.at<cv::Vec3b>(i, j)[2] = 0;
			}
			else
			{
				out.at<cv::Vec3b>(i, j)[0] = 255;
				out.at<cv::Vec3b>(i, j)[1] = 255;
				out.at<cv::Vec3b>(i, j)[2] = 255;
			}
		}

	resize(out, out, med);

	for (int i = 0; i < out.rows; i++)
		for (int j = 0; j < out.cols; j++)
		{
			if (out.at<cv::Vec3b>(i, j)[0] + out.at<cv::Vec3b>(i, j)[1] + out.at<cv::Vec3b>(i, j)[2] < 600)
			{
				out.at<cv::Vec3b>(i, j)[0] = 0;
				out.at<cv::Vec3b>(i, j)[1] = 0;
				out.at<cv::Vec3b>(i, j)[2] = 0;
			}
			else
			{
				out.at<cv::Vec3b>(i, j)[0] = 255;
				out.at<cv::Vec3b>(i, j)[1] = 255;
				out.at<cv::Vec3b>(i, j)[2] = 255;
			}
		}

	resize(out, out, small);

	for (int i = 0; i < out.rows; i++)
		for (int j = 0; j < out.cols; j++)
		{
			if (out.at<cv::Vec3b>(i, j)[0] + out.at<cv::Vec3b>(i, j)[1] + out.at<cv::Vec3b>(i, j)[2] < 600)
			{
				out.at<cv::Vec3b>(i, j)[0] = 0;
				out.at<cv::Vec3b>(i, j)[1] = 0;
				out.at<cv::Vec3b>(i, j)[2] = 0;
			}
			else
			{
				out.at<cv::Vec3b>(i, j)[0] = 255;
				out.at<cv::Vec3b>(i, j)[1] = 255;
				out.at<cv::Vec3b>(i, j)[2] = 255;
			}
		}

	resize(out, out, exsmall);

	for (int i = 0; i < out.rows; i++)
		for (int j = 0; j < out.cols; j++)
		{
			if (out.at<cv::Vec3b>(i, j)[0] + out.at<cv::Vec3b>(i, j)[1] + out.at<cv::Vec3b>(i, j)[2] < 600)
			{
				out.at<cv::Vec3b>(i, j)[0] = 0;
				out.at<cv::Vec3b>(i, j)[1] = 0;
				out.at<cv::Vec3b>(i, j)[2] = 0;
			}
			else
			{
				out.at<cv::Vec3b>(i, j)[0] = 255;
				out.at<cv::Vec3b>(i, j)[1] = 255;
				out.at<cv::Vec3b>(i, j)[2] = 255;
			}
		}

	resize(out, out, standard);

	for (int i = 0; i < out.rows; i++)
		for (int j = 0; j < out.cols; j++)
		{
			if (out.at<cv::Vec3b>(i, j)[0] + out.at<cv::Vec3b>(i, j)[1] + out.at<cv::Vec3b>(i, j)[2] < 600)
			{
				out.at<cv::Vec3b>(i, j)[0] = 0;
				out.at<cv::Vec3b>(i, j)[1] = 0;
				out.at<cv::Vec3b>(i, j)[2] = 0;
			}
			else
			{
				out.at<cv::Vec3b>(i, j)[0] = 255;
				out.at<cv::Vec3b>(i, j)[1] = 255;
				out.at<cv::Vec3b>(i, j)[2] = 255;
			}
		}

	cv::imwrite("inp/img.jpg", out);
}

void cal()
{
	cv::Mat img = cv::imread("Ex.jpg");

	bake(img);

	read("");
	system("cls");

	remove("inp/img.jpg");

	cv::Mat obj(1, 784, CV_64F);
	ifstream in("inp.inp");

	for (int i = 0; i < 784; i++)
	{
		double zz; in >> zz;
		obj.at<double>(0, i) = zz;
	}
 
	int tag = -1; double z = 0; vector<pair<double, int>> res;
	cv::Mat zz = pca.project(obj);
	for (int i = 0; i < cennum; i++)
	{
		double d = 0;
		for (int j = 0; j < vec; j++)
		d += (zz.at<double>(0, j) - cen.at<double>(i, j)) * (zz.at<double>(0, j) - cen.at<double>(i, j));

		res.pb({ d, (int)centag.at<double>(i, 0) });
	}
	std::sort(res.begin(), res.end());

	for (int i = 0; i < 10; i++) ss[i] = 0;

	int cc = 0;
	for (int i = 0; i<10 ; i++) 
	{
		++cc;

		ss[res[i].second] += 1 - (double)i / 10;
		if(ss[res[i].second] * pp[res[i].second] >z)
		{
			z = ss[res[i].second]*pp[res[i].second];
			tag = res[i].second;
		}
	}

	if (tag == -1) cout << "Not detected" << endl; else
	cout << "Detected number: " << tag << "\n";
	//cout << "Correct guess : "; cin >> ans;

	//cv::imwrite("D:/Work/learningmachine/NumRecog/imgset/rawdata/" + itos(ans) + "/" + itos(ans) + "/" + itos(++num[ans]) + ".jpg", img);

	cout << "Press any key to continue";
    cin.get();
}

void masscal()
{
	system("cls");

	path ii = current_path() / ".." / ".." / "imgset" / "baked";
	ii = weakly_canonical(ii);

	int	ccc[10], sss[10];
	for (int i = 0; i < 10; i++) ccc[i] = sss[i] = 0;

	for (int nn = 0; nn < 10; nn++)
	{
		string inp = ii.string() + "/" + itos(nn) + "/";

		for (const auto& entry : directory_iterator(inp))
		{
			++ccc[nn];
			read(entry.path().string());

			cv::Mat obj(1, 784, CV_64F);
			ifstream in("inp.inp");

			for (int i = 0; i < 784; i++)
			{
				double zz; in >> zz;
				obj.at<double>(0, i) = zz;
			}

			int tag = -1; double z = 0; vector<pair<double, int>> res;
			cv::Mat zz = pca.project(obj);
			for (int i = 0; i < cennum; i++)
			{
				double d = 0;
				for (int j = 0; j < vec; j++)
					d += (zz.at<double>(0, j) - cen.at<double>(i, j)) * (zz.at<double>(0, j) - cen.at<double>(i, j));

				res.pb({ d, (int)centag.at<double>(i, 0) });
			}
			std::sort(res.begin(), res.end());

			for (int i = 0; i < 10; i++) ss[i] = 0;

			int cc = 0;
			for (int i = 0; i<10; i++)
			{
				++cc;

				ss[res[i].second]+=1-(double)i/10;
				if (ss[res[i].second] * pp[res[i].second] > z)
				{
					z = ss[res[i].second] * pp[res[i].second];
					tag = res[i].second;
				}
			}

			if (tag == nn) ++sss[nn];
			cout << "\rProcessing number " << nn << " Tag = " << (tag) << " Processed: " << ccc[nn] << "     " << flush;
		}
	}

	system("cls");
	for (int i = 0; i < 10; i++)
		{
		cout << "Number " << i << " accuracy: " << (double)sss[i] / double(ccc[i]) * 100 << "% (" << sss[i] << "/" << ccc[i] << ")\n";
	}

	int totalc = 0, totals = 0;
	for (int i = 0; i < 10; i++)
	{
		totalc += ccc[i];
		totals += sss[i];
	}
	cout << "Overall accuracy: " << (double)totals / double(totalc) * 100 << "% (" << totals << "/" << totalc << ")\n";
	cout << "Press any key to continue";

	cin.get();
	system("cls");
}

void proc()
{
    while (o) {
        system("cls");
        
		cout << "Enter to proceed" << endl;

		cin.get();
        cal();
    }
}

int main()
{
    preproc();

	system("cls");
	cout << "Mass calculation? (y for yes): ";
	char ch; cin >> ch; cin.get();
	if (ch == 'y') masscal();
    proc();

	path exe =current_path()/ ".." / ".." / "Kmean" / "x64" / "Debug" / "Kmean.exe";

	exe = weakly_canonical(exe);

	//system(exe.string().c_str());

    return 0;
}
