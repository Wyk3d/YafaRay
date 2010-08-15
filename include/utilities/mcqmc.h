//---------------------------------------------------------------------
// Monte Carlo & Quasi Monte Carlo stuff
//---------------------------------------------------------------------

#ifndef __MCQMC_H
#define __MCQMC_H

__BEGIN_YAFRAY
// fast incremental Halton sequence generator
// calculation of value must be double prec.
class /*YAFRAYCORE_EXPORT*/ Halton
{
public:
	Halton() {}
	Halton(int base) { setBase(base); }
	void setBase(int base)
	{
		_base = base;
		invBase = 1.0/double(base);
		value = 0;
	}
	~Halton() {}
	void reset() { value=0.0; }
	void setStart(unsigned int i)
	{
		value = 0.0;
		double f, factor;
		f = factor = invBase;
		while (i>0) {
			value += double(i % _base) * factor;
			i /= _base;
			factor *= f;
		}
	}
	PFLOAT getNext() const
	{
		double r = 1.0 - value - 0.0000000001;
		if (invBase < r)
			value += invBase;
		else {
			double hh, h=invBase;
			do {
				hh = h;
				h *= invBase;
			} while (h >= r);
			value += hh + h - 1.0;
		}
		return value;
	}
private:
	unsigned int _base;
	double invBase;
	mutable double value;
};


// fast base-2 van der Corput, Sobel, and Larcher & Pillichshammer sequences,
// all from "Efficient Multidimensional Sampling" by Alexander Keller
#define multRatio (0.00000000023283064365386962890625)
inline PFLOAT RI_vdC(unsigned int bits, unsigned int r=0)
{
	bits = ( bits << 16) | ( bits >> 16);
	bits = ((bits & 0x00ff00ff) << 8) | ((bits & 0xff00ff00) >> 8);
	bits = ((bits & 0x0f0f0f0f) << 4) | ((bits & 0xf0f0f0f0) >> 4);
	bits = ((bits & 0x33333333) << 2) | ((bits & 0xcccccccc) >> 2);
	bits = ((bits & 0x55555555) << 1) | ((bits & 0xaaaaaaaa) >> 1);
	return (PFLOAT)((double)(bits^r) * multRatio);
}

inline PFLOAT RI_S(unsigned int i, unsigned int r=0)
{
	for (unsigned int v=1<<31; i; i>>=1, v^=v>>1)
		if (i & 1) r ^= v;
	return (PFLOAT)((double) r * multRatio);
}

inline PFLOAT RI_LP(unsigned int i, unsigned int r=0)
{
	for (unsigned int v=1<<31; i; i>>=1, v|=v>>1)
		if (i & 1) r ^= v;
	return (PFLOAT)((double)r * multRatio);
}


inline int nextPrime(int lastPrime)
{
	int newPrime = lastPrime + (lastPrime & 1) + 1;
	for (;;) {
		int dv=3;  bool ispr=true;
		while ((ispr) && (dv*dv<=newPrime)) {
			ispr = ((newPrime % dv)!=0);
			dv += 2;
		}
		if (ispr) break;
		newPrime += 2;
	}
	return newPrime;
}

/* The fnv - Fowler/Noll/Vo- hash code
   unrolled for the special case of hashing 32bit unsigned integers
   very easy but fast
   more details on http://www.isthe.com/chongo/tech/comp/fnv/ 
*/

union Fnv32_u
{
	unsigned int in;
	unsigned char out[4];
};

#define FNV1_32_INIT 0x811c9dc5
#define FNV_32_PRIME 0x01000193

inline unsigned int fnv_32a_buf(unsigned int value)
{
	unsigned int hash = FNV1_32_INIT;
	Fnv32_u val;
	val.in = value;

	for (int i = 0; i<4; i++)
	{
		hash ^= val.out[i];
		hash *= FNV_32_PRIME;
	}

	return hash;
}

/* multiply-with-carry generator x(n) = a*x(n-1) + carry mod 2^32.
   period = (a*2^31)-1 */

/* Choose a value for a from this list
   1791398085 1929682203 1683268614 1965537969 1675393560
   1967773755 1517746329 1447497129 1655692410 1606218150
   2051013963 1075433238 1557985959 1781943330 1893513180
   1631296680 2131995753 2083801278 1873196400 1554115554
*/
#define y_a 1791398085
#define y_ah (y_a >> 16)
#define y_al (y_a & 65535)

class random_t
{
	public:
		random_t(): x(30903), c(0) {}
		random_t(unsigned int seed): x(30903), c(seed) {}
		void reset(unsigned int seed) { x = 30903; c = seed; }
		double operator()()
		{
			unsigned int xh = x>>16, xl = x & 65535;
			x = x * y_a + c;
			c = xh*y_ah + ((xh*y_al) >> 16) + ((xl*y_ah) >> 16);
			if (xl*y_al >= ~c + 1) c++;
			return (double)x * multRatio;
		}
	protected:
		unsigned int x, c;
};

#undef y_a
#undef y_ah
#undef y_al

__END_YAFRAY

#endif	//__MCQMC_H
