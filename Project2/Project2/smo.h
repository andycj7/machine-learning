#include "iostream"
#include "fstream"
#include "ctype.h"
#include "math.h"
#include "stdlib.h"
#include "time.h"
using namespace std;
#define N 27//����������
#define end_support_i 27
#define first_test_i 27
#define d 13//ά��
//////////global variables
//int N=     //������
//int d=     //ά��
int C=100;//�ͷ�����
double tolerance=0.001;//ui��߽�0,C֮��Ĺ��Χ
double eps=1e-3;//һ������0��С��
int two_sigma_squared=100;//RBF(Radial-Basis Function)�˺����еĲ�����sigma==(10/2)^1/2��
double alph[end_support_i];//Lagrange multipiers
double b;//threshold
double error_cache[end_support_i];//���non-bound�������
int target[N];//ѵ�������������Ŀ��ֵ
//double precomputed_self_dot_product[N];//Ԥ��dot_product_func(i,i)��ֵ���Լ��ټ�����
double dense_points[N][d];//���ѵ�������������0-end_support_i-1ѵ��;first_test_i-N����

//����������
int takeStep(int,int);
int examineNonBound(int);
int examineBound(int);
int examineFirstChoice(int,double);
int examineExample(int);
double kernel_func(int,int);
double learned_func(int);
double dot_product_func(int,int);
double error_rate();
void setX();//��������ֵ����
void setT();//����Ŀ��ֵ����
void initialize();
///////////////////////////////////////////////////////
