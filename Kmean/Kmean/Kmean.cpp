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

const ll oo = 1e18l + 7;
const ll range = 200;

struct Point {
    vector<long double> cord;
    int tag;

    long double dd;
} mt;

vector<Point> points;
vector<Point> cen;

random_device rd;
const int cennum=200;

int n;
vector<int> idx;

void preproc()
{

    path baker = current_path() / ".." / ".." / "DataBaker" / "x64" / "Debug" / "DataBaker.exe";
    baker = weakly_canonical(baker);
    path reader = current_path() / ".." / ".." / "ImgReader" / "x64" / "Debug" / "ImgReader.exe";
    reader = weakly_canonical(reader);

    system(baker.string().c_str());
    system(reader.string().c_str());
}

int Rand()
{
    mt19937 rand(rd());
    return rand() % (range * 2) - range;
}

void proc()
{
    path in = current_path() / ".." / ".." / "imgset" / "number_input" / "inp.inp";
    in = weakly_canonical(in);

    ifstream inp(in.string().c_str());

    inp >> n;

    cv::Mat zz(n, 28*28, CV_32F);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < 28 * 28; j++)
        {
            int x; inp >> x;
			zz.at<float>(i, j) = x;
			if (zz.at<float>(i, j)) zz.at<float>(i, j) = 1.0f;
        }
        int z; inp >> z; idx.pb(z);
    }

    // 20-40
    cv::PCA pca(zz,cv::Mat(),cv::PCA::DATA_AS_ROW,25);
    cv::Mat eigenvectors = pca.eigenvectors;

    cv::Mat zz_pca,label,cen;
	pca.project(zz, zz_pca);

    cv::kmeans(zz_pca, cennum, label,
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 1000, 0.0001),
		100, cv::KMEANS_PP_CENTERS, cen);

	int tag[cennum+1];
    
    vector<map<int, int>> freq(cennum); 

    for (int i = 0; i < n; i++) {
        int cc = label.at<int>(i, 0);
        freq[cc][idx[i]]++;
    }

    for (int c = 0; c < cennum; c++) {
        int bestLabel = -1, bestCnt = 0;
        for (auto& p : freq[c]) {
            if (p.second > bestCnt) {
                bestCnt = p.second;
                bestLabel = p.first;
            }
        }
        tag[c] = bestLabel;
    }

    int crct = 0;
    for (int i = 0; i < n; i++)
    {
        cout << "Point " << i << ", Kmeans Tag = " << label.at<int>(i, 0) << ", Tag = " << tag[label.at<int>(i, 0)] << ", idx=" <<idx[i] << "\n";
        if (tag[label.at<int>(i, 0)] == idx[i]) crct++;
    }

	cout << "Correctness: " << (long double)crct / n * 100.0l << "%\n";

    path ou = current_path() / ".." / ".." / "imgset" / "cluster.txt";
    ou = weakly_canonical(ou);

    ofstream cout(ou.string().c_str());

    cout << "pca_mean" << pca.mean << endl;
    cout << "pca_eigenvectors" << pca.eigenvectors << endl;
    cout << "pca_eigenvalues" << pca.eigenvalues << endl;

    cout << "centroids" << cen << endl;

    cv::Mat tagMat(cennum, 1, CV_32S);

    for (int i = 0; i < cennum; i++)
        tagMat.at<int>(i, 0) = tag[i];

    cout << "centroid_tags" << cv::Mat(tagMat).reshape(1, cennum);
}

int main()
{
    preproc();
    proc();
    return 0;
}
