#ifndef LPCFUNC_H
#define LPCFUNC_H

//#include <itpp/base/vec.h>
extern "C" {
#define sqr(x)  (x * x)
	//const double dPi = 3.14159265358979323846;
	//#define it_error_if(t,s) if ((t)){std::cout << s << endl;}

	//class Vec 
	//{
	//public:
	//	Vec(int size)
	//	{
	//		if(size > 0)
	//			m_data = new double[size];
	//		m_size = size;
	//	}
	//	int length() const { return m_size; }
	//	double *_data() { return m_data; m_size = -1; }
	//	void clear() { delete[] m_data;  }
	//	double &operator[](int i)
	//	{
	//		if (i < m_size || m_size <= 0)
	//			std::cout << "out of array\n";
	//		return *(m_data + i);
	//	}
	//	const double &operator[](int i) const
	//	{
	//		if (i < m_size)
	//			std::cout << "out of array\n";
	//		return *(m_data + i);
	//	}
	//private:
	//	double *m_data;
	//	int m_size;
	//};
	//	
	//typedef Vec vec;
#define dPi 3.14159265358979323846
	typedef struct DoubleVec
	{
		double *p;
		int size;
	}Vec;
	Vec *createVec(int size);
	void deleteVec(Vec *);

	Vec* autocorr(const Vec *x, int order);
	Vec* levinson(const Vec *R2, int order);
	//double* LPC(const double *x, int size, int order);
	Vec *poly2ac(const Vec *poly);
	Vec *poly2rc(const Vec *a);
	Vec *rc2poly(const Vec *k);
	Vec *rc2lar(const Vec *k);
	Vec *lar2rc(const Vec *LAR);
	double FNevChebP_double(double  x, const double c[], int n);
	double FNevChebP(double  x, const double c[], int n);
	Vec *poly2lsf(const Vec *pc);
	Vec *lsf2poly(const Vec *f);
	Vec *poly2cepstrum(const Vec *a);
	Vec *poly2cepstrumVI(const Vec *a, int num);
	Vec *cepstrum2poly(const Vec *c);
	Vec *chirp(const Vec *a, double factor);
	Vec *schurrc(const Vec *R, int order);
	Vec *lerouxguegenrc(const Vec *R, int order);
}
#endif // #ifndef LPCFUNC_H
