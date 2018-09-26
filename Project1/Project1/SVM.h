#pragma/*
用支持向量机求解二分类问题
分离超平面为：w'·x+b=0
分类决策函数:f(x)=sign(w'·x+b)
*/
#include <iostream>
using namespace std;

class SVM
{
private:
	int sampleNum;	//样本数
	int featureNum;	//特征数
	double **data;	//存放样本 行：样本， 列：特征
	double *label;		//存放类标
	double *alpha;
	//double *w;   对于非线性问题，涉及kernel,不方便算
	double b;
	double *gx;

	double s_max(double, double);
	double s_min(double, double);
	int secondAlpha(int);
	void computeGx();
	double kernel(int, int);
	void update(int, int, double, double);
	bool isConvergence();
	bool takeStep(int, int);

public:
	~SVM();
	//初始化数据
	void initialize(double *, double *, int, int);
	//序列最小最优算法
	void SMO();
	double objFun(int);
	void show();
};
