#include "SVM.h"
#include <math.h>
using namespace std;

#define eps 1e-2	//����
const int C = 100;	//�ͷ�����

SVM::~SVM()
{
	if (data) delete[]data;
	if (label) delete label;
	if (alpha) delete alpha;
	if (gx) delete gx;
}

//d��Ϊ����,ÿ���������д洢; l��ǩ(1��-1); sn��������; fn��������
//alpha ������˹���� ��ʼ��Ϊ0 gx ��ʼ��Ϊ0
void SVM::initialize(double *d, double *l, int sn, int fn)
{
	this->sampleNum = sn;
	this->featureNum = fn;
	this->label = new double[sampleNum];
	this->data = new double*[sampleNum];
	for (int i = 0; i < sampleNum; i++)
	{
		this->label[i] = l[i];
	}
	for (int i = 0; i < sampleNum; i++)
	{
		this->data[i] = new double[featureNum];
		for (int j = 0; j < featureNum; j++)
		{
			data[i][j] = d[i*featureNum + j];
		}
	}
	alpha = new double[sampleNum] {0};
	gx = new double[sampleNum] {0};

}

double SVM::s_max(double a, double b)
{
	return a > b ? a : b;
}

double SVM::s_min(double a, double b)
{
	return a < b ? a : b;
}

double SVM::objFun(int x)
{
	int j = 0;
	//ѡ��һ��0 < alpha[j] < C
	for (int i = 0; i < sampleNum; i++)
	{
		if (alpha[i] > 0 && alpha[i] < C)
		{
			j = i;
			break;
		}
	}
	//����b
	double b = label[j];
	for (int i = 0; i < sampleNum; i++)
	{
		b -= alpha[i] * label[i] * kernel(i, j);
	}
	//������ߺ���
	double objf = b;
	for (int i = 0; i < sampleNum; i++)
	{
		objf += alpha[i] * label[i] * kernel(x, i);
	}
	return objf;
}

//�ж���������
bool SVM::isConvergence()
{
	//alpah[i] * y[i]��͵���0
	//0 <= alpha[i] <= C
	//y[i] * gx[i]����һ������
	double sum = 0;
	for (int i = 0; i < sampleNum; i++)
	{
		if (alpha[i] < -eps || alpha[i] > C + eps) return false;
		else
		{
			// alpha[i] = 0
			if (fabs(alpha[i]) < eps && label[i] * gx[i] < 1 - eps) return false;
			// 0 < alpha[i] < C
			if (alpha[i] > -eps && alpha[i] < C + eps && fabs(label[i] * gx[i] - 1)>eps) return false;
			// alpha[i] = C
			if (fabs(alpha[i] - C) < eps && label[i] * gx[i] > 1 + eps) return false;
		}
		sum += alpha[i] * label[i];
	}
	if (fabs(sum) > eps) return false;

	return true;
}

//��װ�Ǹ��˺���
//�����������ڻ�
double SVM::kernel(int i, int j)
{
	double res = 0;
	for (int k = 0; k < featureNum; k++)
	{
		res += data[i][k] * data[j][k];
	}
	return res;
}

//����g(xi),Ҳ���Ƕ�����i��Ԥ��ֵ
void SVM::computeGx()
{
	for (int i = 0; i < sampleNum; i++)
	{
		gx[i] = 0;
		for (int j = 0; j < sampleNum; j++)
		{
			gx[i] += alpha[j] * label[j] * kernel(i, j);
		}
		gx[i] += b; //������Ԥ��ֵ�����
	}
}

//���ºܶණ��
void SVM::update(int a1, int a2, double x1, double x2)
{
	//������ֵb

	double b1_new = -(gx[a1] - label[a1]) - label[a1] * kernel(a1, a1)*(alpha[a1] - x1)
		- label[a2] * kernel(a2, a1)*(alpha[a2] - x2) + b;
	double b2_new = -(gx[a2] - label[a2]) - label[a1] * kernel(a1, a2)*(alpha[a1] - x1)
		- label[a2] * kernel(a2, a2)*(alpha[a2] - x2) + b;
	if (fabs(alpha[a1]) < eps || fabs(alpha[a1] - C) < eps || fabs(alpha[a2]) < eps || fabs(alpha[a2] - C) < eps)
		b = (b1_new + b2_new) / 2;
	else
		b = b1_new;
	/*
	int j = 0;
	//ѡ��һ��0 < alpha[j] < C
	for (int i = 0; i < sampleNum; i++)
	{
		if (alpha[i]>0 && alpha[i] < C)
		{
			j = i;
			break;
		}
	}
	//����b
	double b = label[j];
	for (int i = 0; i < sampleNum; i++)
	{
		b -= alpha[i] * label[i] * kernel(i, j);
	}
	*/
	//����gx
	computeGx();
}


//ѡȡ�ڶ������� 
/*
��ѡ���Ƕ�ӦE1-E2����
��û�У�������ʽ����ѡĿ�꺯�����㹻�½���alpha2
��û�У�ѡ���µ�alpha1
*/
int SVM::secondAlpha(int a1)
{
	//�ȼ�������е�E��Ҳ��������xi��Ԥ��ֵ����ʵ���֮��Ei=g(xi)-yi
	//��E1Ϊ����ѡ��С��Ei��ΪE2������ѡ���
	bool pos = (gx[a1] - label[a1] > 0);
	double tmp = pos ? 100000000 : -100000000;
	double ei = 0; int a2 = -1;
	for (int i = 0; i < sampleNum; i++)
	{
		ei = gx[i] - label[i];
		if (pos &&  ei < tmp || !pos && ei > tmp)
		{
			tmp = ei;
			a2 = i;
		}
	}
	//�������������ֱ�ӱ�������߽��ϵ�֧��������,ѡ���������½���ֵ
	return a2;
}

//ѡ��a1��a2�����и���
//a1��a2��ѡ���ĵڼ�������
bool SVM::takeStep(int a1, int a2)
{
	if (a1 < -eps) return false;

	double x1, x2;		//old alpha
	x2 = alpha[a2];
	x1 = alpha[a1];
	//��������ı߽�
	double L, H;
	double s = label[a1] * label[a2];//a1 �� a2ͬ�Ż����
	L = s < 0 ? s_max(0, alpha[a2] - alpha[a1]) : s_max(0, alpha[a2] + alpha[a1] - C);
	H = s < 0 ? s_min(C, C + alpha[a2] - alpha[a1]) : s_min(C, alpha[a2] + alpha[a1]);
	if (L >= H) return false;
	double eta = kernel(a1, a1) + kernel(a2, a2) - 2 * kernel(a1, a2);
	//����alpah[a2]
	if (eta > 0)
	{
		alpha[a2] = x2 + label[a2] * (gx[a1] - label[a1] - gx[a2] + label[a2]) / eta;
		if (alpha[a2] < L) alpha[a2] = L;
		else if (alpha[a2] > H) alpha[a2] = H;
	}
	else
    //���κ����������ϵ��eta<=0
	//��ʱ��Ҫ�����Сֵ���ڱ߽���
	{
		alpha[a2] = L;
		//����Ҫ��Ķ��κ�����ֵ
		//����ʵ���е���
		double Lobj = objFun(a2);
		alpha[a2] = H;
		double Hobj = objFun(a2);
		if (Lobj < Hobj - eps)
			alpha[a2] = L;
		else if (Lobj > Hobj + eps)
			alpha[a2] = H;
		else
			alpha[a2] = x2;
	}
	//�½�̫�٣����Բ���
	if (fabs(alpha[a2] - x2) < eps*(alpha[a2] + x2 + eps))
	{
		alpha[a2] = x2;
		return false;
	}
	//����alpha[a1]
	alpha[a1] = x1 + s * (x2 - alpha[a2]);


	update(a1, a2, x1, x2);
	/*
	for (int ii = 0; ii < sampleNum; ii++)
	{
		cout << gx[ii] << endl;
	}
	*/
	return true;

}

//��SVM������ߵĶ�ż���Ż��������alpha
/*
��������С���Ż��㷨��SMO�����alpha
step1:ѡȡһ����Ҫ���µı���alpha[i]��alpha[j]
step2:�̶�alpha[i]��alpha[j]����Ĳ���������ż��������Ż����ø��º��alpha[i]��alpha[j]
�ο������ͳ��ѧϰ������
	 JC Platt��Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines��
*/
void SVM::SMO()
{
	//bool convergence = false;	//�ж���û������
	int a1, a2;
	bool Changed = true;  //��û�и���
	int numChanged = 0;		//�����˶��ٴ�
	int *eligSample = new int[sampleNum]; // ��¼���ʹ�������
	int cnt = 0;				 //��������
	computeGx();

	do
	{
		numChanged = 0; cnt = 0;
		//ѡ���һ�������������KKT�����������㣩
		//����ѡ 0 < alpha < C ������ , alpha�����ź���ĵ��������仯
		for (int i = 0; i < sampleNum; i++)
		{
			//��¼�²�����KKT��������������������
			if (Changed)
			{
				cnt = 0;
				for (int j = 0; j < sampleNum; j++)
				{
					if (alpha[j] > eps && alpha[j] < C - eps)
					{
						eligSample[cnt++] = j;
					}
				}
				Changed = false;
			}
			//0<alpha<C
			if (alpha[i] > eps && alpha[i] < C - eps)
			{
				a1 = i;
				//������KKT������������С��eps��Լ����0
				if (fabs(label[i] * gx[i] - 1) > eps)
				{
					//ѡ��ڶ�������������ѡ�½�����
					a2 = secondAlpha(i);
					Changed = takeStep(a1, a2);
					numChanged += Changed;
					if (Changed) continue;
					else //Ŀ�꺯��û���½�
					{
						//�����α�������߽��ϵ�
						for (int j = 0; j < cnt; j++)
						{

							if (eligSample[j] == i) continue;
							a2 = eligSample[j];
							Changed = takeStep(a1, a2);
							numChanged += Changed;

							if (Changed) break;
						}
						if (Changed) continue;
						//�ٱ����������ݼ�
						int k = 0;
						for (int j = 0; j < sampleNum; j++)
						{
							//���������Ѿ��Թ��ļ���ϵĵ�
							if (eligSample[k] == j)
							{
								k++;
								continue;
							}
							a2 = j;
							Changed = takeStep(a1, a2);
							numChanged += Changed;

							if (Changed) break;
						}
						//�Ҳ������ʵ�alpha2�� ��һ��alpha1
					}
				}
			}

		}
		if (numChanged)//�Ѿ��иı���
		{
			Changed = false;
			continue;
		}
		//ѡ����������KKT����������
		for (int i = 0; i < sampleNum; i++)
		{
			a1 = i;
		    //Υ����KKT����:alpha=C&&y*f(x)-1<0||alpha=0&&y*f(x)-1>0
			//�������ܱ�֤ԭ�������Сֵ
			//�������ĸ���������С������
			if (fabs(alpha[i]) < eps && label[i] * gx[i] < 1 ||
				fabs(alpha[i] - C) < eps && label[i] * gx[i] > 1)
			{
				//ѡ��ڶ�������,����ͬ��
				a2 = secondAlpha(i);
				Changed = takeStep(a1, a2);
				numChanged += Changed;

				if (Changed) continue;
				else //Ŀ�꺯��û���½�
				{
					//�����α�������߽��ϵ�
					//����߽��ϵĵ��Ѿ���¼��eligSample����
					for (int j = 0; j < cnt; j++)
					{

						if (eligSample[j] == i) continue;
						a2 = eligSample[j];
						Changed = takeStep(a1, a2);
						numChanged += Changed;

						if (Changed) break;
					}
					if (Changed) continue;
					//�ٱ����������ݼ�
					int k = 0;
					for (int j = 0; j < sampleNum; j++)
					{
						if (j == eligSample[k])
						{
							k++;
							continue;
						}
						a2 = j;
						Changed = takeStep(a1, a2);
						numChanged += Changed;

						if (Changed) break;
					}
					//�Ҳ������ʵ�alpha2�� ��һ��alpha1
				}
			}
		}
		/*
//		if (!Changed)
		{
			cout<<"num"<<numChanged<<endl;
			show();
		}

		//��ͳ��ѧϰ��������˵�������������������������
		//���Ը���JC Platt����α�������᷽����Ҳ������ȫһ����
		convergence = isConvergence();
		//show();
		cnt++;
		if (cnt == 10000)
		{
			cout << "num" << numChanged << endl;
			show();
		}
		*/
	} while (numChanged);

	delete eligSample;
}

void SVM::show()
{
	cout << "֧������Ϊ:" << endl;
	for (int i = 0; i < sampleNum; i++)
	{
		if (alpha[i] > eps)
			cout << i << " ��Ӧ��alphaΪ:" << alpha[i] << endl;
	}
	cout << endl;
}
