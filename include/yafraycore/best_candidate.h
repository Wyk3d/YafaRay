#ifndef Y_BESTCANDIDATE_H
#define Y_BESTCANDIDATE_H

#include <limits>
#include <core_api/vector3d.h>

__BEGIN_YAFRAY

template <
	class MemberType, 
	class ContainerIteratorType, 
	class FuncHolder,
		MemberType &(FuncHolder::*GetMemberFunc)(typename ContainerIteratorType::reference)
>
class MemberIterator : public std::iterator_traits<const MemberType*>
{
protected:
	ContainerIteratorType container_iter;
	FuncHolder funcHolder;
public:
	MemberIterator(const FuncHolder &funcHolder, const ContainerIteratorType &iter) :
	  container_iter(iter), funcHolder(funcHolder) {}

	  MemberIterator(const MemberIterator &iter) {
		  container_iter = iter->container_iter;
	  }

	  MemberType &operator *() {
		  return ((&funcHolder)->*GetMemberFunc)(typename ContainerIteratorType::reference(*container_iter));
	  }

	  MemberIterator& operator++() {
		  ++container_iter;
		  return *this;
	  }

	  MemberIterator& operator--() {
		  --container_iter;
		  return *this;
	  }

	  bool operator == (const MemberIterator &iter) {
		  return container_iter == iter.container_iter;
	  }

	  bool operator != (const MemberIterator &iter) {
		  return !((*this)==iter);
	  }
};

class BestCandidateSampler {
protected:
	struct cand {
		point3d_t p;
		bool t;
		float min_dist;
	};

	typedef std::vector<cand> CandVecType;
	CandVecType cands;
	int nr_to_generate;
	int nr_to_keep;

	friend class SampleIterator;

	class SampleIterator : public std::iterator_traits<const cand*>
	{
	protected:
		BestCandidateSampler &sampler;
		CandVecType::iterator iter;
	public:

		SampleIterator(BestCandidateSampler &sampler, CandVecType::iterator &iter) 
			: sampler(sampler), iter(iter) {}

		SampleIterator(const SampleIterator &iter)
			: sampler(iter.sampler), iter(iter.iter) {}

		reference operator *() {
			return *iter;
		}

		SampleIterator& operator++() {
			do {
				++iter;
			} while(iter != sampler.cands.end() && !iter->t);
			return *this;
		}

		SampleIterator& operator--() {
			do {
				--iter;
			} while(!iter->t);
			return *this;
		}

		bool operator ==(const SampleIterator &iter) {
			return this->iter == iter.iter;
		}

		bool operator !=(const SampleIterator &iter) {
			return !((*this)==iter);
		}

		void operator = (SampleIterator &iter) {
			this->iter = iter.iter;
			this->sampler = iter.sampler;
		}
	};

	struct GetPointFuncHolder {
		const point3d_t &getPoint(const cand& c) {
			return c.p;
		}
	};

public:

	typedef MemberIterator<const point3d_t, SampleIterator, 
		GetPointFuncHolder, &GetPointFuncHolder::getPoint> iterator;

	iterator begin() {
		CandVecType::iterator itr = cands.begin();
		while(itr != cands.end() && !itr->t)
			++itr;

		return iterator(GetPointFuncHolder(), 
			SampleIterator(*this, itr));
	}

	iterator end() {
		CandVecType::iterator itr = cands.end();
		return iterator(GetPointFuncHolder(), 
			SampleIterator(*this, itr));
	}

	BestCandidateSampler() : nr_to_generate(-1), nr_to_keep(-1) {

	}

	template<class PointGenFunc> 
	void gen_candidates(PointGenFunc gen_functor, int nr_to_keep, int nr_to_generate = -1) {
		this->nr_to_keep = nr_to_keep;
		if(nr_to_generate < 0) {
			nr_to_generate = nr_to_keep * 5;
		}
		this->nr_to_generate = nr_to_generate;

		cands.resize(nr_to_generate);

		printf("generating %d candiates\n", nr_to_generate);

		for(int j = 0; j < nr_to_generate; j++)
		{
			cands[j].p = gen_functor(j);
			cands[j].t = false;
			cands[j].min_dist = std::numeric_limits<float>::max();
		}
	}

	void gen_samples(int nr_to_keep = -1) {
		if(nr_to_keep == -1) {
			nr_to_keep = this->nr_to_keep;
		} else {
			if(nr_to_keep >= nr_to_generate) {
				nr_to_keep = nr_to_generate;
			}
			// TODO: check number already generated and
			// generate the difference
		}

		printf("generating %d samples of %d\n", nr_to_keep, nr_to_generate);


		int ppoz = 0;

		// the current minimum distance from the taken samples
		// to each non-taken candidate is stored in cand.min_dist
		for(int j = 0; j < nr_to_keep; j++)
		{
			point3d_t &p = cands[ppoz].p;
			cands[ppoz].t = true;

			// for each new sample, update the min_dist in the candidates
			// and find the next sample by taking that highest min_dist
			float max_mind = std::numeric_limits<float>::min();
			for(int k = 0; k < nr_to_generate; k++) {
				if(!cands[k].t) {
					float d = (cands[k].p - p).lengthSqr();
					float &mind = cands[k].min_dist;
					if(d < mind) 
						mind = d;

					if(mind > max_mind) {
						max_mind = mind;
						ppoz = k;
					}
				}
			}
		}
	}
};

__END_YAFRAY

#endif // Y_BESTCANDIDATE_H