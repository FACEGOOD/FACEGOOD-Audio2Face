#pragma once
#ifndef MYDLL_EXPORTS
#define MYDLL _declspec(dllexport)
#else
#define MYDLL _declspec(dllimport)
#endif 
extern "C" {
	__declspec(dllexport) int LPC(double *in, int size, int order, double *out);
}