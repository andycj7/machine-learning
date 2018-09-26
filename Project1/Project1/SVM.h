#pragma/*
��֧��������������������
���볬ƽ��Ϊ��w'��x+b=0
������ߺ���:f(x)=sign(w'��x+b)
*/
#include <iostream>
using namespace std;

class SVM
{
private:
	int sampleNum;	//������
	int featureNum;	//������
	double **data;	//������� �У������� �У�����
	double *label;		//������
	double *alpha;
	//double *w;   ���ڷ��������⣬�漰kernel,��������
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
	//��ʼ������
	void initialize(double *, double *, int, int);
	//������С�����㷨
	void SMO();
	double objFun(int);
	void show();
};
