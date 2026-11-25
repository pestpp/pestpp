#ifndef ENIFGRAPH_H_
#define ENIFGRAPH_H_

#include <string>
#include <vector>
#include <fstream>
#include <ostream>
#include <map>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/OrderingMethods>
#include "covariance.h"

using namespace std;

/* A conditional-independence graph over the adjustable parameters, and the
sparse prior precision estimated on it.

This is the input mode the ensemble information filter is actually built around:
rather than supplying a p x p prior covariance - which does not scale - the user
supplies the SPARSITY PATTERN of the prior precision and the precision itself is
estimated from the prior ensemble.  A dense covariance generally has a nearly
sparse inverse, which is the whole premise.

The graph is read with Mat::from_file(), so it comes for free in every matrix
format pest++ already understands:

    .jcb / .jco   binary; the extended/coo variant is genuine (i,j,value) triplet
                  storage with long names, so it stays sparse on disk
    .mat / .cov   ascii PEST matrix format
    .csv          comma separated

Any structural non-zero is an edge; the value is ignored.  The matrix is
symmetrised (G union G^T) and the diagonal forced on, so a thresholded
correlation matrix, a localiser product, or a hand-built adjacency all mean the
same thing.

Note on how the precision is USED: H^T R^-1 H is dense unless H is also sparse,
so forming the posterior precision explicitly would throw the sparsity away.
Instead the solve keeps the woodbury form and obtains C*H^T by sparse-solving
Lam * X = H^T for the n observation right-hand sides.  No p x p dense matrix is
ever formed and no sparsity is required of H.
*/
class EnifGraph
{
public:
	EnifGraph() : initialized(false), nnz_offdiag(0) {}

	/* read the graph and align it to par_names.  throws if the file does not
	cover every adjustable parameter - a silently mis-aligned graph would be
	worse than no graph at all. */
	void from_file(const string& filename, const vector<string>& par_names,
		ofstream& frec);

	/* estimate the prior precision on the graph from ensemble anomalies
	(p x N, already scaled by 1/sqrt(N-1)) by neighbourhood regression:
	for each node i, regress u_i on its graph neighbours, which gives one row
	of the cholesky-like factor.  shrink is a stein-type pull toward the
	diagonal, needed when a node's neighbourhood approaches the ensemble size. */
	void estimate_precision(const Eigen::MatrixXd& anomalies, double shrink,
		ofstream& frec);
	/* read the prior precision itself from a file - values, not just the pattern.
	the same file is the graph, so from_file() is called first with it.  a precision
	that is model-based (from the variogram the prior was drawn from) rather than
	estimated from the ensemble is the one thing the estimated graph path could not
	supply at npar >> N.  the matrix must be symmetric and positive definite. */
	void load_precision(const string& filename, ofstream& frec);
	//shrink < 0 (the default): ridge per node by 5-fold cross-validation over the
	//realizations, k/(N-1) when there are fewer than 20

	/* apply the implied prior covariance: returns C * M, computed as
	solve(Lam, M) so the covariance is never formed */
	Eigen::MatrixXd apply_cov(const Eigen::MatrixXd& M) const;

	/* damped gauss-newton step in the information form, for use when H is sparse:

	    Lam_post = (1+lam) Lam_prior + H^T Rinv H
	    g        = Lam_prior (x - x0) + H^T Rinv (y - d)
	    delta    = -solve(Lam_post, g)

	violating no sparsity: with a sparse H the posterior precision stays sparse and
	this is one sparse cholesky, instead of the n right-hand-side solves woodbury
	needs.  returns the upgrade (p x N). */
	Eigen::MatrixXd information_step(const Eigen::SparseMatrix<double>& H,
		const Eigen::VectorXd& rinv, const Eigen::MatrixXd& e,
		const Eigen::MatrixXd& resid, double lam, ostream& frec) const;

	bool is_initialized() const { return initialized; }
	bool has_precision() const { return prec_ready; }
	int num_nodes() const { return (int)names.size(); }
	int num_edges() const { return nnz_offdiag / 2; }
	const Eigen::SparseMatrix<double>& precision() const { return prec; }
	void report(ofstream& frec) const;

private:
	bool initialized = false;
	bool prec_ready = false;
	int nnz_offdiag;
	vector<string> names;
	/* the ordering the precision is estimated in, and the support of each row of
	the cholesky-like factor.  solve_order lists the original node indices in the
	order they are eliminated; pred_sets[i] holds the original indices node i is
	regressed on, which is the fill-inclusive factor pattern, not just the direct
	graph neighbours.  this is what graphite-maps does: it fits the pattern of the
	cholesky factor of the PERMUTED graph, and the permutation is chosen to keep
	that fill small. */
	vector<int> solve_order;
	vector<vector<int>> pred_sets;
	int fill_edges = 0;
	Eigen::SparseMatrix<double> adj;
	Eigen::SparseMatrix<double> prec;
	mutable Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
};


/* Estimate a SPARSE observation operator H (n x p) by lasso regression of the
observation anomalies on the parameter anomalies, one independent problem per
observation row.

Sparsity here is load-bearing, not cosmetic.  A structural zero in H is the
statement "this observation carries no information about this parameter", which
is what makes H an inspectable influence map.  It is also what keeps
H^T Rinv H - and therefore the posterior precision - sparse, so the graph is
worth having at all.

lasso_frac scales the penalty relative to the smallest value that would zero a
whole row, so it is dimensionless and lives in (0,1).  unexplained returns the
per-observation residual variance, which is the quantity the observation error is
inflated by.  Rows are independent, so this parallelises over num_threads.

unexp_divisor is what the raw residual sum of squares is divided by to get the
unexplained variance.  the default (anything <= 0) is the number of columns,
which is what the mean-centred fit wants.  the realization-centred fit passes
the number of columns that actually carry a deviation (its centre column is
identically zero), or the weight sum when the columns are weighted.

loc, when given, is an observations x parameters pattern: a stored entry says that
observation may use that parameter, anything else is held at zero in H.  each row is
then the lasso over just its listed parameters, penalty scaled by the row-zeroing
value over those parameters too, and a row with none listed stays empty.  only the
pattern is used, not the values.  with loc the lassos run in column order: the outer
loop is over parameters and the inner loop over the rows still iterating.

frec is any ostream, so threads can each write to their own stringstream. */
Eigen::SparseMatrix<double> estimate_sparse_H(const Eigen::MatrixXd& A,
	const Eigen::MatrixXd& B, double lasso_frac, int num_threads,
	Eigen::VectorXd& unexplained, ostream& frec,
	double unexp_divisor = -1.0,
	const Eigen::SparseMatrix<double>* loc = nullptr);

/* the enif update with an explicit sparse H and the supplied prior covariance C,
by woodbury, so no p x p inverse is formed:

    C_lam = C / (1 + lam),   G = diag(noise_var) + H C_lam H^T
    delta = -[ (e - C_lam H^T G^-1 H e) / (1 + lam) + C_lam H^T G^-1 resid ]

this is the same step solve_enif takes on its covariance path, with H given
instead of carried implicitly as B M A^T.  e and resid are p x N and n x N, so
one column gives one realization.  noise_var is the per-observation noise
variance, already inflated if that is wanted. */
Eigen::MatrixXd enif_woodbury_step(const Eigen::SparseMatrix<double>& H,
	const Eigen::VectorXd& noise_var, const Eigen::SparseMatrix<double>& C,
	const Eigen::MatrixXd& e, const Eigen::MatrixXd& resid, double lam);

/* the same step when H C H^T has rank N (the ensemble size) and is given as a
factor, so nothing sized n x n or p x n is ever formed:

    H C H^T = U Q U^T       U is n x N, Q is N x N symmetric and at least psd
    C H^T   = L U^T         L is p x N
    G = diag(noise_var) + s U Q U^T,  s = 1 / (1 + lam)
    delta = -s [ e + L U^T G^-1 (resid - s He) ]

G^-1 is applied by woodbury a second time, in the form that needs no inverse
of Q (Q may be singular):

    G^-1 v = Rinv v - Rinv U (sQ) (I + U^T Rinv U sQ)^-1 U^T Rinv v

the two cases that use it: the implicit ensemble H = B M A^T, where
U = B, Q = M A^T C A M and L = C A M, for any C; and an explicit sparse H with
the current ensemble covariance C = A A^T, where U = H A, Q = I and L = A.
He = H e is passed in because only the caller knows how to apply H.  every
matrix here is N columns wide apart from the n x N and p x N ones, so the
cost is (n + p) N^2 and the memory is the anomalies themselves. */
Eigen::MatrixXd enif_lowrank_step(const Eigen::MatrixXd& U, const Eigen::MatrixXd& Q,
	const Eigen::MatrixXd& L, const Eigen::VectorXd& noise_var,
	const Eigen::MatrixXd& e, const Eigen::MatrixXd& He, const Eigen::MatrixXd& resid, double lam);

/* per observation group summary of the noise inflation */
struct EnifInflateGroupStats
{
	int count = 0;
	double noise_var = 0.0;     //mean of 1/w^2
	double unexp_var = 0.0;     //mean of the variance H fails to explain
	double ratio_mean = 0.0;    //mean of (noise_var + unexp_var) / noise_var per obs
	double ratio_min = 0.0;
	double ratio_max = 0.0;
	double weight_mean = 0.0;   //mean of the control file weight
	double eff_weight_mean = 0.0; //mean of the weight the update actually used
};

/* report what the unexplained-variance inflation did to each observation.  the
noise variance the update uses is 1/w^2 + unexplained (when applied), so the
effective weight is 1/sqrt(that).  writes one row per observation to
csv_filename and a group summary to the rec, sorted by mean inflation ratio,
in the same shape as the group phi summary.  the mean unexplained variance
that used to be the only thing reported mixes units across groups, so it is
useless whenever the groups are in different units - hence per group.
returns the group stats so the selftest can check the arithmetic. */
map<string, EnifInflateGroupStats> enif_inflation_report(
	const vector<string>& obs_names, const vector<string>& groups,
	const Eigen::VectorXd& weights, const Eigen::VectorXd& unexplained,
	bool applied, int iter, const string& csv_filename, ofstream& frec);

/* the per-realization version of that report, for the multimodal enif solve where
every realization has its own H, its own weights and its own unexplained variance.
one row per (realization, observation) in long form: real_name, obs_name, group,
weight, noise_var, unexplained_var, inflated_var, inflate_ratio, effective_weight
and h_row_nnz (how many parameters that realization's H row touches; 0 means the
lasso left the row empty and the observation is all noise for that realization).
weights, unexplained and H are aligned with real_names.  returns the mean inflate
ratio per realization, which is what the rec summary and the selftest want. */
map<string, double> enif_mm_inflation_csv(const vector<string>& real_names,
	const vector<string>& obs_names, const vector<string>& groups,
	const vector<Eigen::VectorXd>& weights, const vector<Eigen::VectorXd>& unexplained,
	const vector<const Eigen::SparseMatrix<double>*>& H, bool applied,
	const string& csv_filename);

#endif // ENIFGRAPH_H_
