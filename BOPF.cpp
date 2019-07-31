//Copyright (c) 2017 Xiaosheng Li (xli22@gmu.edu)
//Reference: Linear Time Complexity Time Series Classification with Bag-of-Pattern-Features ICDM 2017
/*
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include<vector>
#include <cstring>
#include <string>
#include <fstream>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#define INF 1e20
#define MAX_PER_LINE 100000
#define ROUNDNUM(x) ((int)(x + 0.5))

using namespace std;

void loadDataset(char* trainFileName, vector<vector<double> > &train, vector<double> &trainLabel) {
	FILE *trainp;
	char buf[MAX_PER_LINE];
	char* tmp;
	double label;
	trainp = fopen(trainFileName, "r");

	if (trainp == NULL) {
		cout << "Error!! Input file " << trainFileName << " not found." << endl;
		exit(0);
	}

	while (fgets(buf, MAX_PER_LINE, trainp)) {
		vector<double> ts;
		tmp = strtok(buf, ", \r\n");
		label = atof(tmp);
		trainLabel.push_back(label);
		tmp = strtok(NULL, ", \r\n");
		while (tmp != NULL) {
			ts.push_back(atof(tmp));
			tmp = strtok(NULL, ", \r\n");
		}
		train.push_back(ts);
	};
	fclose(trainp);
}

vector<int> adjustLabel(const vector<double> &trainLabel, vector<double> &trainLabelIndex, vector<double> &tlabel) {
	vector<double>::const_iterator vit;
	sort(tlabel.begin(), tlabel.end());
	tlabel.erase(unique(tlabel.begin(), tlabel.end()), tlabel.end());
	trainLabelIndex.reserve(trainLabel.size());
	vector<int> classCount(tlabel.size(), 0);

	for (vit = trainLabel.cbegin(); vit != trainLabel.cend(); ++vit) {
		int position = find(tlabel.begin(), tlabel.end(), *vit) - tlabel.begin();
		trainLabelIndex.push_back(position);
		classCount[position]++;
	}
	return classCount;
}

int* adjustLabelSet(const double* trainLabel, double* trainLabelIndex, vector<double> &tlabel, int m) {
	unordered_set<double> labelset;
	int i;
	for (i = 0; i < m; ++i) {
		labelset.insert(trainLabel[i]);
	}

	tlabel.assign(labelset.begin(), labelset.end());
	sort(tlabel.begin(), tlabel.end());
	int* classCount = (int *)malloc(sizeof(int) * (tlabel.size()));
	fill(classCount, classCount + tlabel.size(), 0);

	for (i = 0; i < m; ++i) {
		const int position = find(tlabel.begin(), tlabel.end(), trainLabel[i]) - tlabel.begin();
		trainLabelIndex[i] = position;
		classCount[position]++;
	}
	return classCount;
}

void cumsum(double* train, double* cumTrain, double* cumTrain2, int m, int n) {
	int i, j;
	double* pt = train;
	double* pcx = cumTrain;
	double* pcx2 = cumTrain2;
	for (i = 0; i < m; ++i) {
		*(pcx++) = 0;
		*(pcx2++) = 0;
	}
	double* sum;
	sum = (double *)malloc(sizeof(double) * (m));
	double* psum = sum;
	for (i = 0; i < m; ++i) {
		*(psum++) = 0;
	}
	double* sum2;
	sum2 = (double *)malloc(sizeof(double) * (m));
	double* psum2 = sum2;
	for (i = 0; i < m; ++i) {
		*(psum2++) = 0;
	}

	for (j = 0; j < n; ++j) {
		psum = sum;
		psum2 = sum2;
		for (i = 0; i < m; ++i) {
			*psum += *pt;
			*psum2 += (*pt)*(*pt);
			*(pcx++) = *psum;
			*(pcx2++) = *psum2;
			psum++;
			psum2++;
			pt++;
		}
	}
	free(sum);
	free(sum2);
}

void bop(const double* train, int wd, int wl, const double* cumTrain, const double* cumTrain2, int m, int n, int* train_bop) {
	const double* pmat = train;
	const double* pcx = cumTrain;
	const double* pcx2 = cumTrain2;
	int i, j, k, pword, u, l;
	const int bopsize = 1 << (2 * wd);
	const double ns = (1.0 * wl) / wd;
	int* pr = train_bop;
	for (i = 0; i < m*bopsize; ++i) {
		*(pr++) = 0;
	}
	pr = train_bop;

	for (i = 0; i<m; ++i) {
		pword = -1;
		for (j = 0; j<n - wl + 1; ++j) {
			const double sumx = *(pcx + i + (j + wl)*m) - *(pcx + i + j*m);
			const double sumx2 = *(pcx2 + i + (j + wl)*m) - *(pcx2 + i + j*m);
			const double meanx = sumx / wl;
			const double sigmax = sqrt(sumx2 / wl - meanx*meanx);
			int wordp = 0;
			for (k = 0; k<wd; ++k) {
				u = ROUNDNUM(ns*(k + 1));
				l = ROUNDNUM(ns*k);
				const double sumsub = *(pcx + i + (j + u)*m) - *(pcx + i + (j + l)*m);
				const double avgsub = sumsub / (u - l);
				const double paa = (avgsub - meanx) / sigmax;
				int val;
				if (paa < 0)
					if (paa < -0.67) val = 0;
					else val = 1;
				else
					if (paa < 0.67) val = 2;
					else val = 3;
					const int ws = (1 << (2 * k))*val;
					wordp += ws;
			}
			if (pword != wordp) {
				(*(pr + i + wordp*m)) += 1;
				pword = wordp;
			}
		}
	}
}

void anova(const int* train_bop, const double* trainLabelIndex, const int* classCount, int m, int n, double* bop_f_a, int c) {
	const int* pmat = train_bop;
	const double* plb = trainLabelIndex;
	int i, j, k;
	double* pr;
	pr = bop_f_a;

	double sumall;
	double avgall;
	double *sumc = (double *)malloc(sizeof(double) * (c));
	double *avgc = (double *)malloc(sizeof(double) * (c));
	double ssa;
	double ssw;
	double msa;
	double msw;

	for (j = 0; j<n; j++) {
		sumall = 0.0;
		for (k = 0; k<c; k++) {
			sumc[k] = 0.0;
		}
		for (i = 0; i<m; i++) {
			k = (int)*(plb + i);
			sumall += *(pmat + i + j*m);
			sumc[k] += *(pmat + i + j*m);
		}
		avgall = sumall / m;
		ssa = 0.0;
		for (k = 0; k<c; k++) {
			avgc[k] = sumc[k] / classCount[k];
			ssa += classCount[k] * (avgc[k] - avgall)*(avgc[k] - avgall);
		}
		ssw = 0.0;
		for (i = 0; i<m; i++) {
			k = (int)*(plb + i);
			ssw += (*(pmat + i + j*m) - avgc[k])*(*(pmat + i + j*m) - avgc[k]);
		}
		msa = ssa / (c - 1);
		msw = ssw / (m - c);
		if (msa == 0 && msw == 0)
			*(pr++) = 0;
		else
			*(pr++) = msa / msw;
	}
	free(sumc);
	free(avgc);
}

struct Cmp {
	double* values;
	Cmp(double* vec) : values(vec){}
	bool operator() (const int& a, const int& b) const{
		return values[a] > values[b];
	}
};

double* crossVL(int* train_bop, const double * trainLabelIndex, const int* classCount, pair<int, double> &feaNumAcc, int m, int n, int c) {
	int* pmat = train_bop;
	const double* plb = trainLabelIndex;
	int* pm;
	int i, j, k, p, maxk;
	double rMin, r, label, d, count, maxcount, countc;
	double* crossL;
	crossL = (double *)malloc(sizeof(double) * (c*n));
	double* pr = crossL;

	double **x = (double **)malloc(sizeof(double*) * c);
	double **y = (double **)malloc(sizeof(double*) * m);
	for (i = 0; i<c; i++) {
		x[i] = (double *)malloc(sizeof(double) * n);
	}
	for (i = 0; i<c; i++) {
		for (j = 0; j<n; j++) {
			x[i][j] = 0.0;
		}
	}
	for (i = 0; i<m; i++) {
		y[i] = (double *)malloc(sizeof(double) * c);
	}
	for (i = 0; i<m; i++) {
		for (j = 0; j<c; j++) {
			y[i][j] = 0.0;
		}
	}

	maxcount = -1;
	for (k = 0; k<n; k++) {
		count = 0;
		for (i = 0; i<m; i++) {
			p = (int)*(plb + i);
			x[p][k] += *(pmat + i + k*m);
		}
		for (i = 0; i<c; i++) {
			x[i][k] = x[i][k] / classCount[i];
			*(pr++) = x[i][k];
		}
		for (i = 0; i<m; i++) {
			rMin = INF;
			label = 0.0;
			p = (int)*(plb + i);
			countc = classCount[p];
			for (j = 0; j<c; j++) {
				r = y[i][j];
				pm = pmat + i + k*m;
				d = *pm - x[j][k];
				if (j == p) d += (*pm) / countc;
				r += d*d;
				y[i][j] = r;
				if (r<rMin) {
					rMin = r;
					label = j;
				}
			}
			if (label == *(plb + i)) count++;
		}
		if (count >= maxcount) {
			maxcount = count;
			maxk = k;
		}
	}

	feaNumAcc.first = maxk + 1;
	feaNumAcc.second = maxcount / m;

	for (i = 0; i<m; i++){
		free(y[i]);
	}
	free(y);
	for (i = 0; i<c; i++){
		free(x[i]);
	}
	free(x);
	return crossL;
}

double* crossVL2(int* train_bop, const double * trainLabelIndex, const int* classCount, pair<int, double> &feaNumAcc, int m, int n, int c) {
	int* pmat = train_bop;
	const double* plb = trainLabelIndex;
	int i, j, k, p, maxk;
	double rMax, r, label, d, count, maxcount, r1, r2, r3, countc;
	double* crossL;
	crossL = (double *)malloc(sizeof(double) * (c*n));
	double* pr = crossL;

	double **x;
	double **y;
	double **y2;
	double **y3;
	x = (double **)malloc(sizeof(double*) * c);
	for (i = 0; i<c; i++) {
		x[i] = (double *)malloc(sizeof(double) * n);
	}
	for (i = 0; i<c; i++) {
		for (j = 0; j<n; j++) {
			x[i][j] = 0.0;
		}
	}
	y = (double **)malloc(sizeof(double*) * m);
	for (i = 0; i<m; i++) {
		y[i] = (double *)malloc(sizeof(double) * c);
	}
	for (i = 0; i<m; i++) {
		for (j = 0; j<c; j++) {
			y[i][j] = 0.0;
		}
	}
	y2 = (double **)malloc(sizeof(double*) * m);
	for (i = 0; i<m; i++) {
		y2[i] = (double *)malloc(sizeof(double) * c);
	}
	for (i = 0; i<m; i++) {
		for (j = 0; j<c; j++) {
			y2[i][j] = 0.0;
		}
	}
	y3 = (double **)malloc(sizeof(double*) * m);
	for (i = 0; i<m; i++) {
		y3[i] = (double *)malloc(sizeof(double) * c);
	}
	for (i = 0; i<m; i++) {
		for (j = 0; j<c; j++) {
			y3[i][j] = 0.0;
		}
	}

	maxcount = -1;
	for (k = 0; k<n; k++) {
		count = 0;
		for (i = 0; i<m; i++) {
			p = (int)*(plb + i);
			x[p][k] += *(pmat + i + k*m);
		}
		countc = 0.0;
		for (i = 0; i<c; i++) {
			if (x[i][k]>0) countc++;
		}
		for (i = 0; i<c; i++) {
			if (x[i][k]>0) x[i][k] = (1 + log10(x[i][k]))*(log10(1 + c / countc));
			*(pr++) = x[i][k];
		}
		for (i = 0; i<m; i++) {
			rMax = -INF;
			label = 0.0;
			for (j = 0; j<c; j++) {
				r1 = y[i][j];
				r2 = y2[i][j];
				r3 = y3[i][j];
				d = *(pmat + i + k*m);
				if (d>0) d = 1 + log10(d);
				r1 += d * x[j][k];
				r2 += d*d;
				r3 += x[j][k] * x[j][k];
				y[i][j] = r1;
				y2[i][j] = r2;
				y3[i][j] = r3;
				r = r1*r1 / (r2*r3);
				if (r>rMax) {
					rMax = r;
					label = j;
				}
			}
			if (label == *(plb + i)) count++;
		}
		if (count >= maxcount) {
			maxcount = count;
			maxk = k;
		}
	}
	feaNumAcc.first = maxk + 1;
	feaNumAcc.second = maxcount / m;

	for (i = 0; i<m; i++){
		free(y[i]);
	}
	free(y);
	for (i = 0; i<m; i++){
		free(y2[i]);
	}
	free(y2);
	for (i = 0; i<m; i++){
		free(y3[i]);
	}
	free(y3);
	for (i = 0; i<c; i++){
		free(x[i]);
	}
	free(x);
	return crossL;
}

double* trimArr(double* ps, int c, int fn){
	double* pd = (double *)malloc(sizeof(double) * (c*fn));
	memcpy(pd, ps, c*fn*sizeof(double));
	return pd;
}

int* trimArr2(int* sortIndex, int n) {
	int* pd = (int *)malloc(sizeof(int) * n);
	int* ps = sortIndex;
	memcpy(pd, ps, n*sizeof(int));
	return pd;
}

void classify(int* test_bop, double* train_bop_centroid, const vector<double> &tlabel, int p, int m, int n, double* H) {
	int* ptmat = test_bop;
	double* pmat = train_bop_centroid;
	int* pm;
	double* pv;
	int i, j, k;
	double r, rMin, d, label;
	double* h = H;

	for (i = 0; i<p; i++) {
		rMin = INF;
		for (j = 0; j<m; j++) {
			r = 0.0;
			pm = ptmat + i;
			pv = pmat + j;
			for (k = 0; k<n; k++) {
				d = *pm - *pv;
				pm += p;
				pv += m;
				r += d*d;
			}
			if (r<rMin) {
				rMin = r;
				label = tlabel[j];
			}
		}
		*(h + i) = label;
	}
}

void classify2(int* test_bop, double* train_bop_centroid, const vector<double> &tlabel, int p, int m, int n, double* H) {
	int* ptmat = test_bop;
	double* pmat = train_bop_centroid;
	double* pr = H;
	int* pm;
	double* pv;
	int i, j, k;
	double r, rMax, d, label, r1, r2, r3;
	for (i = 0; i<p; i++) {
		rMax = -INF;
		for (j = 0; j<m; j++) {
			r1 = 0.0;
			r2 = 0.0;
			r3 = 0.0;
			pm = ptmat + i;
			pv = pmat + j;
			for (k = 0; k<n; k++) {
				d = *pm;
				if (d>0) d = 1 + log10(d);
				r1 += d*(*pv);
				r2 += d*d;
				r3 += (*pv)*(*pv);
				pm += p;
				pv += m;
			}
			r = r1*r1 / (r2*r3);
			if (r>rMax) {
				rMax = r;
				label = tlabel[j];
			}
		}
		*(pr++) = label;
	}
}

double* mode(const double* H, int n, int m) {
	int i, j, maxcount, count;
	double d, maxnum;

	double* h = (double *)malloc(sizeof(double) * n);
	unordered_map<double, int> ::const_iterator it;
	unordered_map<double, int> map;
	for (j = 0; j < n; j++) {
		map.clear();
		map.reserve(m);
		for (i = 0; i < m; i++) {
			d = *(H + j + i*n);
			if (map.find(d) != map.end()) {
				map[d]++;
			}
			else {
				map.insert(make_pair<double&, int>(d, 1));
			}
		}
		maxcount = -1;
		for (it = map.cbegin(); it != map.cend(); ++it) {
			count = it->second;
			if (count > maxcount) {
				maxcount = count;
				maxnum = it->first;
			}
		}
		*(h + j) = maxnum;
	}
	return h;
}

double calError(const double* h, const double* testLabel, int m) {
	double errorRate;
	int i, count;
	count = 0;
	for (i = 0; i < m; i++) {
		if (h[i] != testLabel[i]) count++;
	}
	errorRate = 1.0 * count / m;
	return errorRate;
}

int* sortArrIndex(double* bop_f_a, int bopsize) {
	int i;
	int* sortIndex = (int *)malloc(sizeof(int) * bopsize);
	int * pr = sortIndex;
	for (i = 0; i < bopsize; ++i) {
		*(pr++) = i;
	}
	sort(sortIndex, sortIndex + bopsize, Cmp(bop_f_a));
	return sortIndex;
}

int countNonZero(int* sortIndex, double *bop_f_a, int bopsize) {
	int i, j;
	i = 0;
	j = bopsize - 1;
	while (bop_f_a[sortIndex[i]] != 0 && i < j) {
		i++;
	}
	return i;
}

void sortTrimArr(const int* train_bop, const int* sortIndex, int m, int n, int* train_bop_sort) {
	int i, j, k;
	int* pt = train_bop_sort;
	for (j = 0; j < n; ++j) {
		k = sortIndex[j];
		for (i = 0; i < m; ++i) {
			*(pt++) = *(train_bop + i + k*m);
		}
	}
}

void vecs2arr(const vector<vector<double> > &trainV, double* train) {
	int i, j, m, n;
	double *pt = train;
	m = trainV.size();
	n = trainV[0].size();
	for (j = 0; j < n; j++) {
		for (i = 0; i < m; i++) {
			*(pt++) = trainV[i][j];
		}
	}
}

void vec2arr(const vector<double> &trainLabelV, double* trainLabel) {
	int i, m;
	double *pt = trainLabel;
	m = trainLabelV.size();
	for (i = 0; i < m; i++) {
		*(pt++) = trainLabelV[i];
	}
}

double sortAvgAcc(const vector<double> cvAcc, const vector<int> index, int m) {
	int i;
	double sumAcc = 0;
	for (i = 0; i < m; i++) {
		sumAcc += cvAcc[index[i]];
	}
	return sumAcc / m;
}

double* mergeArr(const double* H, const double* H2, int m, int n, int n2) {
	int i;
	double* H12 = (double *)malloc(sizeof(double) * m * (n + n2));
	double* pt = H12;
	for (i = 0; i < m*n; i++) {
		*(pt++) = H[i];
	}
	for (i = 0; i < m*n2; i++) {
		*(pt++) = H2[i];
	}
	return H12;
}

int main(int argc, char *argv[]) {

	string dataset = string(argv[1]);
	//string path = "./UCR_TS_Archive_2015/";
	//string path2 = "./UCR_TS_Archive_2015/";
	string path = "./";
	string path2 = "./";
	char trainFileName[200];
	char testFileName[200];
	strcpy(trainFileName, path.append(dataset).append("/").append(dataset).append("_TRAIN").c_str());
	strcpy(testFileName, path2.append(dataset).append("/").append(dataset).append("_TEST").c_str());

	cout << "dataset:" << dataset << endl;

	int m, n, i, j, wl, wd, k, wdIndex, wlIndex, seIndex, mt, nt, c;
	double tStart, tEnd, tStart2, tEnd2;
	vector<vector<double> > trainV;
	vector<double> trainLabelV;
	vector<vector<double> > testV;
	vector<double> testLabelV;

	//load the dataset specified by the input argument
	loadDataset(trainFileName, trainV, trainLabelV);
	loadDataset(testFileName, testV, testLabelV);
	n = trainV[0].size();
	m = trainV.size();
	nt = testV[0].size();
	mt = testV.size();

	double* train = (double *)malloc(sizeof(double) * (m*n));
	double* trainLabel = (double *)malloc(sizeof(double) * m);
	double* test = (double *)malloc(sizeof(double) * (mt*nt));
	double* testLabel = (double *)malloc(sizeof(double) * mt);

	//transform the vectors to arrays
	vecs2arr(trainV, train);
	vec2arr(trainLabelV, trainLabel);
	vecs2arr(testV, test);
	vec2arr(testLabelV, testLabel);

	tStart = clock();

	vector<double> tlabel;
	double* trainLabelIndex = (double *)malloc(sizeof(double) * m);
	int* classCount = adjustLabelSet(trainLabel, trainLabelIndex, tlabel, m);
	c = tlabel.size();

	double* cumTrain = (double *)malloc(sizeof(double) * (m*(n + 1)));
	double* cumTrain2 = (double *)malloc(sizeof(double) * (m*(n + 1)));
	double* cumTest = (double *)malloc(sizeof(double) * (mt*(nt + 1)));
	double* cumTest2 = (double *)malloc(sizeof(double) * (mt*(nt + 1)));

	//calculate the cumulative sums
	cumsum(train, cumTrain, cumTrain2, m, n);
	cumsum(test, cumTest, cumTest2, mt, nt);

	//wd set in the paper
	int wdArray[] = { 3, 4, 5, 6, 7 };
	vector<int> wdList(wdArray, wdArray + 5);

	//wl set in the paper
	vector<int> wlList;
	wlList.reserve(40);
	for (i = 0; i < 40; ++i) {
		wl = ROUNDNUM((i + 1)*0.025*n);
		if (wl >= 10) wlList.push_back(wl);
	}
	wlList.erase(unique(wlList.begin(), wlList.end()), wlList.end());

	int wdNum = wdList.size();
	int wlNum = wlList.size();

	int totalNum = wdNum * wlNum;
	vector<int> bop_feaNum(totalNum, 0);
	vector<double> cvAcc(totalNum, 0);
	vector<int> bop_feaNum2(totalNum, 0);
	vector<double> cvAcc2(totalNum, 0);

	double** bop_feature = (double **)malloc(sizeof(double*) * totalNum);
	int** bop_feature_index = (int **)malloc(sizeof(int*) * totalNum);
	double** bop_feature2 = (double **)malloc(sizeof(double*) * totalNum);
	int** bop_feature_index2 = (int **)malloc(sizeof(int*) * totalNum);

	pair<int, double> feaNumAcc(1, 0);
	pair<int, double> feaNumAcc2(1, 0);
	int maxbopsize = (1 << (2 * wdList[wdNum - 1]));
	int* train_bop = (int *)malloc(sizeof(int) * (m*maxbopsize));
	double* bop_f_a = (double *)malloc(sizeof(double) * maxbopsize);
	int* train_bop_sort = (int *)malloc(sizeof(int) * (m*maxbopsize));

	//grid seach on the parameter combinations
	for (i = 0; i < wdNum; ++i) {
		wd = wdList[i];
		const int bopsize = (1 << (2 * wd));
		for (j = 0; j < wlNum; ++j) {
			wl = wlList[j];
			const int pos = i * wlNum + j;
			//transform time series to BOP representations
			bop(train, wd, wl, cumTrain, cumTrain2, m, n, train_bop);

			//calcuate the ANOVA F values for the features
			anova(train_bop, trainLabelIndex, classCount, m, bopsize, bop_f_a, c);
			int* sortIndex = sortArrIndex(bop_f_a, bopsize);
			int feaNum = countNonZero(sortIndex, bop_f_a, bopsize);

			//select the non-zero F value features
			sortTrimArr(train_bop, sortIndex, m, feaNum, train_bop_sort);

			//perform the cross validation process based on centroids
			double* bop_centroid = crossVL(train_bop_sort, trainLabelIndex, classCount, feaNumAcc, m, feaNum, c);

			//perform the cross validation process based on tf-idf
			double* bop_centroid2 = crossVL2(train_bop_sort, trainLabelIndex, classCount, feaNumAcc2, m, feaNum, c);

			//store the results for testing
			bop_feature[pos] = trimArr(bop_centroid, c, feaNumAcc.first);
			bop_feaNum[pos] = feaNumAcc.first;
			cvAcc[pos] = feaNumAcc.second;
			bop_feature_index[pos] = trimArr2(sortIndex, feaNumAcc.first);
			bop_feature2[pos] = trimArr(bop_centroid2, c, feaNumAcc2.first);
			bop_feaNum2[pos] = feaNumAcc2.first;
			cvAcc2[pos] = feaNumAcc2.second;
			bop_feature_index2[pos] = trimArr2(sortIndex, feaNumAcc2.first);

			free(bop_centroid);
			free(bop_centroid2);
			free(sortIndex);
		}
	}

	free(train_bop);
	free(bop_f_a);
	free(train_bop_sort);

	tEnd = clock();

	tStart2 = clock();

	vector<int> index(totalNum, 0);
	for (k = 0; k < totalNum; ++k) {
		index[k] = k;
	}
	stable_sort(index.begin(), index.end(), Cmp(&cvAcc[0]));

	vector<int> index2(totalNum, 0);
	for (k = 0; k < totalNum; ++k) {
		index2[k] = k;
	}
	stable_sort(index2.begin(), index2.end(), Cmp(&cvAcc2[0]));

	//use the top 30 combinations for testing
	int seNum = 30;
	int seNum2 = 30;
	double* H = (double *)malloc(sizeof(double) * (mt*seNum));
	double* H2 = (double *)malloc(sizeof(double) * (mt*seNum2));

	int* test_bop = (int *)malloc(sizeof(int) * (mt*maxbopsize));
	int* test_bop_sort = (int *)malloc(sizeof(int) * (mt*maxbopsize));

	double avgAcc1 = sortAvgAcc(cvAcc, index, seNum);
	double avgAcc2 = sortAvgAcc(cvAcc2, index2, seNum2);
	double maxAcc = avgAcc1 > avgAcc2 ? avgAcc1 : avgAcc2;

	//apply1 indicates using centroid and apply2 indicates using tf-idf
	bool apply1 = avgAcc1 > 0.7*maxAcc;
	bool apply2 = avgAcc2 > 0.7*maxAcc;

	//testing phase
	if (apply1) {
		for (i = 0; i < seNum; ++i) {
			seIndex = index[i];
			wdIndex = (int)floor(1.0*seIndex / wlNum);
			wlIndex = seIndex % wlNum;
			wd = wdList[wdIndex];
			wl = wlList[wlIndex];

			//transform test series to BOP representatoin
			bop(test, wd, wl, cumTest, cumTest2, mt, nt, test_bop);

			//use the features from training to test
			int* sortIndex = bop_feature_index[seIndex];
			sortTrimArr(test_bop, sortIndex, mt, bop_feaNum[seIndex], test_bop_sort);
			double* train_bop_centroid = bop_feature[seIndex];

			//classify by comparing the test features to the learned centroids and if-idfs
			classify(test_bop_sort, train_bop_centroid, tlabel, mt, c, bop_feaNum[seIndex], H + i*mt);
		}
	}

	if (apply2) {
		for (i = 0; i < seNum2; ++i) {
			seIndex = index2[i];
			wdIndex = (int)floor(1.0*seIndex / wlNum);
			wlIndex = seIndex % wlNum;
			wd = wdList[wdIndex];
			wl = wlList[wlIndex];
			bop(test, wd, wl, cumTest, cumTest2, mt, nt, test_bop);
			int* sortIndex = bop_feature_index2[seIndex];
			sortTrimArr(test_bop, sortIndex, mt, bop_feaNum2[seIndex], test_bop_sort);
			double* train_bop_centroid = bop_feature2[seIndex];
			classify2(test_bop_sort, train_bop_centroid, tlabel, mt, c, bop_feaNum2[seIndex], H2 + i*mt);
		}
	}

	//majority vote of H to produce final prediction h
	double* h;
	if (apply1 && apply2) {
		double* H12 = mergeArr(H, H2, mt, seNum, seNum2);
		h = mode(H12, mt, seNum + seNum2);
		free(H12);
	}
	else if (apply1) {
		h = mode(H, mt, seNum);
	}
	else {
		h = mode(H2, mt, seNum2);
	}

	//calculate the testing error rate
	double errorRate = calError(h, testLabel, mt);
		
	tEnd2 = clock();

	cout << "The training time is " << fixed << (tEnd - tStart) / CLOCKS_PER_SEC << " seconds"<<endl;
	cout << "The testing time is " << fixed << (tEnd2 - tStart2) / CLOCKS_PER_SEC << " seconds"<<endl;
	cout.unsetf(ostream::floatfield);
	cout << "The testing error rate is " << errorRate << endl;
		
	free(train);
	free(trainLabel);
	free(test);
	free(testLabel);
	free(trainLabelIndex);
	free(cumTrain);
	free(cumTrain2);
	free(cumTest);
	free(cumTest2);
	for (i = 0; i<totalNum; i++){
		free(bop_feature[i]);
	}
	free(bop_feature);
	for (i = 0; i<totalNum; i++){
		free(bop_feature_index[i]);
	}
	free(bop_feature_index);
	for (i = 0; i<totalNum; i++){
		free(bop_feature2[i]);
	}
	free(bop_feature2);
	for (i = 0; i<totalNum; i++){
		free(bop_feature_index2[i]);
	}
	free(bop_feature_index2);
	free(test_bop);
	free(test_bop_sort);
	free(H);
	free(H2);
	free(h);
	free(classCount);

	return 0;
}
