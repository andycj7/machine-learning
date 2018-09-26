#include <Windows.h>
#include "SVM.h"
#include "matrix.h"
#include "mat.h"
#include <iostream>
#pragma comment(lib, "libmat.lib")
#pragma comment(lib, "libmx.lib")
using namespace std;

const int fn = 13;
const int sn1 = 59;
const int sn2 = 71;
const int sn3 = 48;
const int sn = 178;

int readData(double* &data, double* &label)
{
	MATFile *pmatFile = NULL;
	mxArray *pdata = NULL;
	mxArray *plabel = NULL;

	int ndir;//������Ŀ
			 //��ȡ�����ļ�
	pmatFile = matOpen("wine_data.mat", "r");
	if (pmatFile == NULL) return -1;
	/*��ȡ.mat�ļ��о��������
	char **c = matGetDir(pmatFile, &ndir);
	if (c == NULL) return -2;
	*/
	pdata = matGetVariable(pmatFile, "wine_data");
	data = (double *)mxGetData(pdata);
	matClose(pmatFile);
	//��ȡ���
	pmatFile = matOpen("wine_label.mat", "r");
	if (pmatFile == NULL) return -1;
	plabel = matGetVariable(pmatFile, "wine_label");

	label = (double *)mxGetData(plabel);
	matClose(pmatFile);

}

int main()
{
	double *data;
	double *label;
	readData(data, label);

	//��Ҫע���.mat�ļ��ж�ȡ�������ݰ��д洢
	double *d;
	double *l;
	SVM svm;

	//��һ�����ݼ���ڶ������ݼ� Ԥ����
	l = new double[sn1 + sn2];
	for (int i = 0; i < sn1 + sn2; i++)
	{
		if (fabs(label[i] - 2) < 1e-3) l[i] = -1;
		else l[i] = 1;
	}
	d = new double[(sn1 + sn2)*fn];
	for (int i = 0; i < fn; i++)
	{
		for (int j = 0; j < sn1 + sn2; j++)
		{
			d[j*fn + i] = data[i*sn + j];
		}
	}
	/*
	for (int i = 0; i < sn1 + sn2; i++)
	{
		for (int j = 0; j < fn; j++)
		{
			cout << d[i*fn + j] << ' ';
		}
		cout << endl;
	}
	*/
	svm.initialize(d, l, sn1 + sn2, fn);
	svm.SMO();
	cout << "���ݼ�1�����ݼ�2";
	svm.show();
	delete l;
	delete d;

	//�ڶ������ݼ�����������ݼ�
	l = new double[sn2 + sn3];
	for (int i = sn1; i < sn1 + sn2 + sn3; i++)
	{
		if (fabs(label[i] - 2) < 1e-3) l[i - sn1] = 1;
		else if (fabs(label[i] - 3) < 1e-3) l[i - sn1] = -1;
	}
	d = new double[(sn2 + sn3)*fn];
	for (int i = 0; i < fn; i++)
	{
		for (int j = sn1; j < sn; j++)
		{
			d[(j - sn1)*fn + i] = data[i*sn + j];
		}
	}

	svm.initialize(d, l, sn2 + sn3, fn);
	svm.SMO();
	cout << "\n���ݼ�2�����ݼ�3";
	svm.show();
	delete l;
	delete d;

	//��һ�����ݼ��͵��������ݼ�
	l = new double[sn1 + sn3];
	for (int i = 0; i < sn1 + sn2 + sn3; i++)
	{
		if (fabs(label[i] - 1) < 1e-3) l[i] = 1;
		else if (fabs(label[i] - 3) < 1e-3) l[i - sn2] = -1;
	}
	d = new double[(sn1 + sn3)*fn];
	for (int i = 0; i < fn; i++)
	{
		for (int j = 0; j < sn1; j++)
		{
			d[j*fn + i] = data[i*sn + j];
		}
		for (int j = sn1 + sn2; j < sn; j++)
		{
			d[(j - sn2)*fn + i] = data[i*sn + j];
		}
	}

	svm.initialize(d, l, sn1 + sn3, fn);
	svm.SMO();
	cout << "\n���ݼ�1�����ݼ�3";
	svm.show();
	delete l;
	delete d;
	getchar();
	return 0;
}
