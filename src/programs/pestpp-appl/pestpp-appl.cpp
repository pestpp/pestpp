// pestpp-appl
//
// Applies the multiplier / addend parameter files that pyemu's PstFrom sets up to the model input
// files, at model run time. It reads the table PstFrom writes ("mult2model_info.csv") and, for each
// model input file listed there, reads the saved copy of the original file (the "org" file), applies
// each multiplier file with the row's operator, enforces the ultimate bounds and writes the model
// input file. It is a compiled equivalent of pyemu.helpers.apply_list_and_array_pars().
//
// Two kinds of model input file are handled:
//   array files - a 2-D block of numbers (rows with an empty "index_cols" entry)
//   list files  - a table whose columns can hold text, integers and reals (rows with "index_cols")
//
// Rows with a "pp_file" entry are pilot points: the multiplier array is first made from the pilot
// point values and a text file of kriging factors ("fac_file"), as pyemu.geostats.fac2real does.
//
// What it does NOT do: run the Python strings in the "pre_apply_function"/"post_apply_function"
// columns, or read the binary factor files pypestutils writes. The caller does those and then runs
// this program for one "apply_order" value at a time (--apply_order).
//
// The program is deliberately self-contained (no PEST++ libraries) so that it starts quickly; it is
// run once per model run.

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "config_os.h"

using namespace std;

namespace
{

// std::from_chars for floating point is missing from older standard libraries (libc++ before 20,
// libstdc++ before 11). strtod is the fallback there: correct, but it takes a process-wide lock on
// some platforms, so threads then give less speed-up.
// apple's libc++ also has it but only allows it when targeting macos 26, so with an older
// deployment target (the release and wheel builds) the availability flag turns it off.
// (define APPL_NO_FP_FROM_CHARS to force the fallback, for testing)
#if defined(APPL_NO_FP_FROM_CHARS)
#elif defined(_LIBCPP_VERSION)
#if _LIBCPP_VERSION >= 200000 && (!defined(_LIBCPP_AVAILABILITY_HAS_FROM_CHARS_FLOATING_POINT) || _LIBCPP_AVAILABILITY_HAS_FROM_CHARS_FLOATING_POINT)
#define APPL_HAVE_FP_FROM_CHARS 1
#endif
#elif defined(__cpp_lib_to_chars) || defined(_MSC_VER)
#define APPL_HAVE_FP_FROM_CHARS 1
#endif

[[noreturn]] void fail(const string& msg) { throw runtime_error(msg); }

// ---------------------------------------------------------------------------------------------
// small text helpers
// ---------------------------------------------------------------------------------------------

string read_file(const string& path)
{
	FILE* f = fopen(path.c_str(), "rb");
	if (!f) fail("cannot open file '" + path + "'");
	string buf;
	char chunk[1 << 16];
	if (fseek(f, 0, SEEK_END) == 0)
	{
		long n = ftell(f);
		if (n > 0) buf.reserve((size_t)n);
	}
	clearerr(f);
	if (fseek(f, 0, SEEK_SET) != 0)
	{
		fclose(f);
		fail("error reading file '" + path + "'");
	}
	size_t got;
	while ((got = fread(chunk, 1, sizeof(chunk), f)) > 0) buf.append(chunk, got);
	bool bad = ferror(f) != 0;
	fclose(f);
	if (bad) fail("error reading file '" + path + "'");
	return buf;
}

void write_file(const string& path, const string& content)
{
	FILE* f = fopen(path.c_str(), "wb");
	if (!f) fail("cannot open file '" + path + "' for writing");
	size_t put = fwrite(content.data(), 1, content.size(), f);
	bool bad = put != content.size();
	if (fclose(f) != 0) bad = true;
	if (bad) fail("error writing file '" + path + "'");
}

string_view trim(string_view s)
{
	while (!s.empty() && (s.front() == ' ' || s.front() == '\t' || s.front() == '\r' || s.front() == '\n')) s.remove_prefix(1);
	while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r' || s.back() == '\n')) s.remove_suffix(1);
	return s;
}

bool ends_with(const string& s, const char* suffix)
{
	size_t n = strlen(suffix);
	if (s.size() < n) return false;
	for (size_t i = 0; i < n; ++i)
		if (tolower((unsigned char)s[s.size() - n + i]) != suffix[i]) return false;
	return true;
}

// Parse one real number starting at b. Returns the position after it, or nullptr if there is none.
const char* parse_double(const char* b, const char* e, double& v)
{
	if (b < e && *b == '+') ++b;
#ifdef APPL_HAVE_FP_FROM_CHARS
	auto r = from_chars(b, e, v);
	if (r.ec == errc::result_out_of_range)
	{
		// overflow or underflow: let strtod decide between inf and 0
		string tmp(b, r.ptr);
		v = strtod(tmp.c_str(), nullptr);
		return r.ptr;
	}
	if (r.ec != errc()) return nullptr;
	return r.ptr;
#else
	// copy the token so that strtod sees a terminated string
	const char* p = b;
	while (p < e && !(*p == ' ' || *p == '\t' || *p == ',' || *p == '\n' || *p == '\r')) ++p;
	char small[64];
	string big;
	const char* tok;
	size_t n = (size_t)(p - b);
	if (n < sizeof(small)) { memcpy(small, b, n); small[n] = 0; tok = small; }
	else { big.assign(b, n); tok = big.c_str(); }
	char* endp = nullptr;
	v = strtod(tok, &endp);
	if (endp == tok) return nullptr;
	return b + (endp - tok);
#endif
}

// True if the whole of s (ignoring surrounding blanks) is one number.
bool whole_token_number(string_view s, double& v)
{
	s = trim(s);
	if (s.empty()) return false;
	const char* r = parse_double(s.data(), s.data() + s.size(), v);
	return r == s.data() + s.size();
}

bool is_whole(double v) { return std::isfinite(v) && v == std::floor(v) && std::fabs(v) < 9.0e15; }

void append_int(string& out, long long v)
{
	char b[24];
	auto r = to_chars(b, b + sizeof(b), v);
	out.append(b, r.ptr - b);
}

// A real number the way Python's repr() prints it (the shortest text that reads back exactly);
// this is what pandas writes for a float in a csv file.
void append_py_repr(string& out, double v)
{
	if (std::isnan(v)) { out += "nan"; return; }
	if (std::isinf(v)) { out += v < 0 ? "-inf" : "inf"; return; }
	char d[48];
	auto r = to_chars(d, d + sizeof(d) - 1, v, chars_format::scientific); // [-]d[.ddd]e[+-]XX
	*r.ptr = 0;
	const char* p = d;
	if (*p == '-') { out.push_back('-'); ++p; }
	const char* epos = strchr(p, 'e');
	int ex = atoi(epos + 1);
	char dig[32];
	int nd = 0;
	for (const char* q = p; q < epos; ++q)
		if (*q != '.') dig[nd++] = *q;
	if (ex >= -4 && ex < 16)
	{
		if (ex < 0)
		{
			out += "0.";
			out.append((size_t)(-ex - 1), '0');
			out.append(dig, nd);
		}
		else
		{
			for (int z = 0; z <= ex; ++z) out.push_back(z < nd ? dig[z] : '0');
			out.push_back('.');
			if (nd > ex + 1) out.append(dig + ex + 1, nd - ex - 1);
			else out.push_back('0');
		}
	}
	else
	{
		out.push_back(dig[0]);
		if (nd > 1) { out.push_back('.'); out.append(dig + 1, nd - 1); }
		out.push_back('e');
		out.push_back(ex < 0 ? '-' : '+');
		int a = abs(ex);
		if (a < 10) out.push_back('0');
		append_int(out, a);
	}
}

// ---------------------------------------------------------------------------------------------
// csv records (comma separated, double-quote aware)
// ---------------------------------------------------------------------------------------------

// Reads one record starting at p and leaves p at the start of the next. Fields are views into the
// buffer, except quoted fields holding a doubled quote, which are un-escaped into `arena`.
void next_csv_record(const char*& p, const char* e, vector<string_view>& fields, deque<string>& arena)
{
	fields.clear();
	while (true)
	{
		if (p < e && *p == '"')
		{
			++p;
			const char* s = p;
			bool doubled = false;
			while (p < e)
			{
				if (*p == '"')
				{
					if (p + 1 < e && p[1] == '"') { doubled = true; p += 2; continue; }
					break;
				}
				++p;
			}
			if (doubled)
			{
				string u;
				for (const char* q = s; q < p; ++q)
				{
					u.push_back(*q);
					if (*q == '"') ++q;
				}
				arena.push_back(std::move(u));
				fields.emplace_back(arena.back());
			}
			else
				fields.emplace_back(s, p - s);
			if (p < e) ++p; // closing quote
			while (p < e && *p != ',' && *p != '\n') ++p;
		}
		else
		{
			const char* s = p;
			while (p < e && *p != ',' && *p != '\n') ++p;
			const char* fe = p;
			if (fe > s && fe[-1] == '\r' && (p >= e || *p == '\n')) --fe;
			fields.emplace_back(s, fe - s);
		}
		if (p >= e) return;
		if (*p == '\n') { ++p; return; }
		++p; // the comma
	}
}

bool record_is_blank(const vector<string_view>& f) { return f.size() == 1 && trim(f[0]).empty(); }

// A field written the way pandas.to_csv writes it: quoted only if it has to be.
void append_csv_field(string& out, string_view f, char sep)
{
	bool q = false;
	for (char c : f)
		if (c == sep || c == '"' || c == '\n' || c == '\r') { q = true; break; }
	if (!q) { out.append(f.data(), f.size()); return; }
	out.push_back('"');
	for (char c : f)
	{
		if (c == '"') out.push_back('"');
		out.push_back(c);
	}
	out.push_back('"');
}

// ---------------------------------------------------------------------------------------------
// the Python list text PstFrom writes in some cells, e.g. "['k', 'i']", "[0, 1, 2]", "[1e+30, None]"
// ---------------------------------------------------------------------------------------------

struct PyItem
{
	bool is_text = false;
	bool is_none = false;
	string text;    // the item as text (for a number: the text as written)
	double num = 0; // the value, when not text and not None
};

vector<PyItem> parse_py_list(const string& cell, const string& what)
{
	vector<PyItem> items;
	string_view s = trim(cell);
	if (s.empty()) return items;
	if (s.front() == '[' || s.front() == '(')
	{
		char close = s.front() == '[' ? ']' : ')';
		if (s.back() != close) fail("cannot read " + what + " entry '" + cell + "'");
		s = s.substr(1, s.size() - 2);
	}
	size_t i = 0;
	while (true)
	{
		while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
		if (i >= s.size()) break;
		PyItem it;
		if (s[i] == '\'' || s[i] == '"')
		{
			char q = s[i++];
			it.is_text = true;
			while (i < s.size() && s[i] != q)
			{
				if (s[i] == '\\' && i + 1 < s.size()) ++i;
				it.text.push_back(s[i++]);
			}
			if (i >= s.size()) fail("cannot read " + what + " entry '" + cell + "'");
			++i;
		}
		else
		{
			size_t j = i;
			int depth = 0;
			while (j < s.size() && (s[j] != ',' || depth > 0))
			{
				if (s[j] == '(') ++depth;
				if (s[j] == ')') --depth;
				++j;
			}
			string_view tok = trim(s.substr(i, j - i));
			// numpy scalars can be written as e.g. np.int64(3)
			size_t lp = tok.find('(');
			if (lp != string_view::npos && !tok.empty() && tok.back() == ')') tok = trim(tok.substr(lp + 1, tok.size() - lp - 2));
			it.text = string(tok);
			if (tok == "None" || tok == "nan" || tok == "NaN") it.is_none = true;
			else if (tok == "True") it.num = 1;
			else if (tok == "False") it.num = 0;
			else if (!whole_token_number(tok, it.num)) fail("cannot read " + what + " entry '" + cell + "'");
			i = j;
		}
		items.push_back(std::move(it));
		while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
		if (i < s.size() && s[i] == ',') ++i;
	}
	return items;
}

// ---------------------------------------------------------------------------------------------
// printf-style format strings (the "fmt" column), applied the way numpy.savetxt applies them
// ---------------------------------------------------------------------------------------------

struct FmtSpec
{
	string before; // literal text in front of the conversion
	bool left = false, plus = false, space = false, zero = false, alt = false;
	int width = 0;
	int prec = -1;
	char conv = 0;
};

struct Fmt
{
	vector<FmtSpec> specs;
	string tail; // literal text after the last conversion
};

Fmt parse_fmt(const string& s)
{
	Fmt f;
	string lit;
	size_t i = 0;
	while (i < s.size())
	{
		if (s[i] != '%') { lit.push_back(s[i++]); continue; }
		if (i + 1 < s.size() && s[i + 1] == '%') { lit.push_back('%'); i += 2; continue; }
		FmtSpec sp;
		sp.before.swap(lit);
		++i;
		for (; i < s.size(); ++i)
		{
			if (s[i] == '-') sp.left = true;
			else if (s[i] == '+') sp.plus = true;
			else if (s[i] == ' ') sp.space = true;
			else if (s[i] == '0') sp.zero = true;
			else if (s[i] == '#') sp.alt = true;
			else break;
		}
		while (i < s.size() && isdigit((unsigned char)s[i])) sp.width = sp.width * 10 + (s[i++] - '0');
		if (i < s.size() && s[i] == '.')
		{
			++i;
			sp.prec = 0;
			while (i < s.size() && isdigit((unsigned char)s[i])) sp.prec = sp.prec * 10 + (s[i++] - '0');
		}
		while (i < s.size() && (s[i] == 'l' || s[i] == 'h' || s[i] == 'L')) ++i;
		if (i >= s.size() || !strchr("eEfFgGdiusr", s[i])) fail("unsupported format string '" + s + "'");
		sp.conv = s[i++];
		f.specs.push_back(std::move(sp));
	}
	f.tail = lit;
	if (f.specs.empty()) fail("format string '" + s + "' has no % conversion");
	return f;
}

void pad_into(string& out, const FmtSpec& sp, const char* body, size_t n, bool numeric)
{
	// body may start with a sign
	if ((int)n >= sp.width) { out.append(body, n); return; }
	size_t pad = sp.width - n;
	if (sp.left) { out.append(body, n); out.append(pad, ' '); }
	else if (sp.zero && numeric)
	{
		size_t s = (n > 0 && (body[0] == '-' || body[0] == '+' || body[0] == ' ')) ? 1 : 0;
		out.append(body, s);
		out.append(pad, '0');
		out.append(body + s, n - s);
	}
	else { out.append(pad, ' '); out.append(body, n); }
}

void format_text(string& out, const FmtSpec& sp, string_view text)
{
	if (sp.conv != 's' && sp.conv != 'r') fail("format conversion '%" + string(1, sp.conv) + "' used on the text value '" + string(text) + "'");
	if (sp.prec >= 0 && (size_t)sp.prec < text.size()) text = text.substr(0, sp.prec);
	pad_into(out, sp, text.data(), text.size(), false);
}

void format_number(string& out, const FmtSpec& sp, double v, bool as_int)
{
	char buf[512];
	size_t n = 0;
	char c = sp.conv;
	if (c == 's' || c == 'r')
	{
		string t;
		if (as_int && is_whole(v)) append_int(t, (long long)v);
		else append_py_repr(t, v);
		format_text(out, sp, t);
		return;
	}
	auto sign_prefix = [&](bool negative) {
		if (negative) buf[n++] = '-';
		else if (sp.plus) buf[n++] = '+';
		else if (sp.space) buf[n++] = ' ';
	};
	if (c == 'd' || c == 'i' || c == 'u')
	{
		if (!std::isfinite(v)) fail("cannot write a non-finite value with an integer format");
		double t = std::trunc(v);
		if (std::fabs(t) >= 9.2e18) fail("value too large for an integer format");
		long long iv = (long long)t;
		sign_prefix(iv < 0);
		unsigned long long mag = iv < 0 ? 0ULL - (unsigned long long)iv : (unsigned long long)iv;
		auto r = to_chars(buf + n, buf + sizeof(buf), mag);
		n = r.ptr - buf;
		pad_into(out, sp, buf, n, true);
		return;
	}
	bool upper = c == 'E' || c == 'F' || c == 'G';
	if (!std::isfinite(v))
	{
		sign_prefix(std::signbit(v) && !std::isnan(v));
		const char* t = std::isnan(v) ? (upper ? "NAN" : "nan") : (upper ? "INF" : "inf");
		memcpy(buf + n, t, 3);
		n += 3;
		pad_into(out, sp, buf, n, false);
		return;
	}
	int prec = sp.prec < 0 ? 6 : sp.prec;
	chars_format cf = (c == 'e' || c == 'E') ? chars_format::scientific : (c == 'f' || c == 'F') ? chars_format::fixed : chars_format::general;
	bool done = false;
	if (!sp.alt)
	{
		sign_prefix(std::signbit(v));
		auto r = to_chars(buf + n, buf + sizeof(buf), std::fabs(v), cf, prec);
		if (r.ec == errc())
		{
			if (upper)
				for (char* q = buf + n; q < r.ptr; ++q)
					if (*q == 'e') *q = 'E';
			n = r.ptr - buf;
			done = true;
		}
	}
	if (!done)
	{
		// rare cases (the '#' flag, or text too long for the buffer): let the C library do it
		string cfmt = "%";
		if (sp.plus) cfmt += '+';
		if (sp.space) cfmt += ' ';
		if (sp.alt) cfmt += '#';
		cfmt += "." + to_string(prec) + string(1, c);
		int len = snprintf(nullptr, 0, cfmt.c_str(), v);
		string big((size_t)len + 1, '\0');
		snprintf(&big[0], big.size(), cfmt.c_str(), v);
		pad_into(out, sp, big.data(), (size_t)len, true);
		return;
	}
	pad_into(out, sp, buf, n, true);
}

// ---------------------------------------------------------------------------------------------
// arrays
// ---------------------------------------------------------------------------------------------

struct Array
{
	vector<double> v;
	size_t nrow = 0, ncol = 0;
	bool one_dim() const { return nrow == 1 || ncol == 1; }
};

// A text block of numbers, read like numpy.loadtxt(ndmin=2): the first `skip` lines are set aside,
// blank lines are ignored and '#' starts a comment.
void parse_text_array(const string& buf, const string& path, size_t skip, Array& a, string* skipped)
{
	const char* p = buf.data();
	const char* e = p + buf.size();
	for (size_t i = 0; i < skip && p < e; ++i)
	{
		const char* nl = (const char*)memchr(p, '\n', e - p);
		p = nl ? nl + 1 : e;
	}
	if (skipped) skipped->assign(buf.data(), p - buf.data());
	a.v.clear();
	a.v.reserve((size_t)(e - p) / 8);
	a.nrow = a.ncol = 0;
	size_t c = 0;
	auto end_row = [&]() {
		if (c == 0) return;
		if (a.nrow == 0) a.ncol = c;
		else if (c != a.ncol) fail("the number of values per line changes at data line " + to_string(a.nrow + 1) + " of '" + path + "'");
		++a.nrow;
		c = 0;
	};
	while (p < e)
	{
		char ch = *p;
		if (ch == ' ' || ch == '\t' || ch == '\r' || ch == ',') { ++p; continue; }
		if (ch == '\n') { end_row(); ++p; continue; }
		if (ch == '#')
		{
			while (p < e && *p != '\n') ++p;
			continue;
		}
		double d;
		const char* r = parse_double(p, e, d);
		if (!r) fail("cannot read a number at data line " + to_string(a.nrow + 1) + " of '" + path + "'");
		a.v.push_back(d);
		++c;
		p = r;
	}
	end_row();
	if (a.v.empty()) fail("no values found in '" + path + "'");
}

// A numpy .npy file holding a 1-D or 2-D array of reals or integers.
void parse_npy(const string& buf, const string& path, Array& a)
{
	if (buf.size() < 12 || memcmp(buf.data(), "\x93NUMPY", 6) != 0) fail("'" + path + "' is not a numpy .npy file");
	size_t hlen, off;
	if (buf[6] == 1) { hlen = (unsigned char)buf[8] | ((size_t)(unsigned char)buf[9] << 8); off = 10; }
	else
	{
		uint32_t h;
		memcpy(&h, buf.data() + 8, 4);
		hlen = h;
		off = 12;
	}
	if (off + hlen > buf.size()) fail("'" + path + "' is truncated");
	string hdr(buf.data() + off, hlen);
	auto value_of = [&](const char* key) -> string {
		size_t k = hdr.find(key);
		if (k == string::npos) fail("cannot read the header of '" + path + "'");
		k = hdr.find(':', k);
		if (k == string::npos) fail("cannot read the header of '" + path + "'");
		size_t s = k + 1;
		while (s < hdr.size() && hdr[s] == ' ') ++s;
		if (s >= hdr.size()) fail("cannot read the header of '" + path + "'");
		size_t t = s;
		if (hdr[s] == '(') t = hdr.find(')', s) + 1;
		else if (hdr[s] == '\'') t = hdr.find('\'', s + 1) + 1;
		else
			while (t < hdr.size() && hdr[t] != ',' && hdr[t] != '}') ++t;
		return hdr.substr(s, t - s);
	};
	string descr = value_of("'descr'");
	bool fortran = value_of("'fortran_order'").find("True") != string::npos;
	vector<PyItem> shape = parse_py_list(value_of("'shape'"), "shape");
	if (shape.empty() || shape.size() > 2) fail("'" + path + "' must hold a 1-D or 2-D array");
	if (descr.size() >= 2) descr = descr.substr(1, descr.size() - 2); // drop the quotes
	if (descr.empty() || descr[0] == '>') fail("'" + path + "': unsupported data type '" + descr + "'");
	string kind = descr.substr(1);
	size_t item = kind == "f8" || kind == "i8" ? 8 : kind == "f4" || kind == "i4" ? 4 : 0;
	if (!item) fail("'" + path + "': unsupported data type '" + descr + "'");
	size_t d0 = (size_t)shape[0].num, d1 = shape.size() == 2 ? (size_t)shape[1].num : 1;
	size_t n = d0 * d1;
	const char* data = buf.data() + off + hlen;
	if (off + hlen + n * item > buf.size()) fail("'" + path + "' is truncated");
	a.v.resize(n);
	if (kind == "f8") memcpy(a.v.data(), data, n * 8);
	else
		for (size_t i = 0; i < n; ++i)
		{
			if (kind == "f4") { float x; memcpy(&x, data + i * 4, 4); a.v[i] = x; }
			else if (kind == "i4") { int32_t x; memcpy(&x, data + i * 4, 4); a.v[i] = x; }
			else { int64_t x; memcpy(&x, data + i * 8, 8); a.v[i] = (double)x; }
		}
	a.nrow = d0;
	a.ncol = d1; // a 1-D array is treated as a single column
	if (fortran && shape.size() == 2)
	{
		vector<double> t(n);
		for (size_t r = 0; r < d0; ++r)
			for (size_t c = 0; c < d1; ++c) t[r * d1 + c] = a.v[c * d0 + r];
		a.v.swap(t);
	}
}

void load_array(const string& path, size_t skip, Array& a, string* skipped)
{
	string buf = read_file(path);
	if (ends_with(path, ".npy"))
	{
		parse_npy(buf, path, a);
		if (skipped) skipped->clear();
	}
	else
		parse_text_array(buf, path, skip, a, skipped);
}

// Arrays used by more than one model input file (a multiplier array shared by many stress periods,
// a zone array, ...) are read once and shared between threads.
class ArrayCache
{
public:
	void set_shared(set<string> names) { shared_names = std::move(names); }
	shared_ptr<const Array> get(const string& path)
	{
		if (!shared_names.count(path))
		{
			auto a = make_shared<Array>();
			load_array(path, 0, *a, nullptr);
			return a;
		}
		shared_ptr<Entry> en;
		{
			lock_guard<mutex> g(mtx);
			auto& slot = entries[path];
			if (!slot) slot = make_shared<Entry>();
			en = slot;
		}
		lock_guard<mutex> g(en->mtx);
		if (!en->arr)
		{
			auto a = make_shared<Array>();
			load_array(path, 0, *a, nullptr);
			en->arr = a;
		}
		return en->arr;
	}

private:
	struct Entry
	{
		mutex mtx;
		shared_ptr<const Array> arr;
	};
	set<string> shared_names;
	mutex mtx;
	map<string, shared_ptr<Entry>> entries;
};

// ---------------------------------------------------------------------------------------------
// the table PstFrom writes (mult2model_info.csv)
// ---------------------------------------------------------------------------------------------

struct Info
{
	vector<string> cols;
	vector<vector<string>> rows;
	unordered_map<string, int> col_index;

	bool has(const string& name) const { return col_index.count(name) > 0; }
	// the cell as text; empty if the column is absent or the cell is empty (pandas "NaN")
	const string& get(size_t row, const string& name) const
	{
		static const string empty;
		auto it = col_index.find(name);
		if (it == col_index.end() || (size_t)it->second >= rows[row].size()) return empty;
		return rows[row][it->second];
	}
};

Info read_info(const string& path)
{
	string buf = read_file(path);
	Info info;
	const char* p = buf.data();
	const char* e = p + buf.size();
	vector<string_view> f;
	deque<string> arena;
	bool first = true;
	while (p < e)
	{
		next_csv_record(p, e, f, arena);
		if (record_is_blank(f)) continue;
		if (first)
		{
			for (auto& x : f) info.cols.emplace_back(x);
			for (size_t i = 0; i < info.cols.size(); ++i) info.col_index[info.cols[i]] = (int)i;
			first = false;
			continue;
		}
		vector<string> row;
		for (auto& x : f) row.emplace_back(x);
		info.rows.push_back(std::move(row));
	}
	if (first) fail("'" + path + "' is empty");
	if (!info.has("model_file") || !info.has("org_file")) fail("'" + path + "' needs 'model_file' and 'org_file' columns");
	// file names may have been written on another operating system
	for (size_t c = 0; c < info.cols.size(); ++c)
		if (info.cols[c].find("file") != string::npos)
			for (auto& row : info.rows)
				if (c < row.size()) replace(row[c].begin(), row[c].end(), '\\', '/');
	return info;
}

bool operator_is_multiply(const string& op, const string& mlt_file)
{
	if (op.empty() || op == "*" || tolower((unsigned char)op[0]) == 'm') return true;
	if (op == "+" || tolower((unsigned char)op[0]) == 'a') return false;
	fail("unrecognized operator '" + op + "' for multiplier file '" + mlt_file + "'");
}

double cell_number(const string& cell, const string& what)
{
	double v;
	if (!whole_token_number(cell, v)) fail("cannot read " + what + " value '" + cell + "'");
	return v;
}

string unique_org_file(const Info& info, const vector<size_t>& rows, const string& model_file)
{
	string org = info.get(rows[0], "org_file");
	for (size_t r : rows)
		if (info.get(r, "org_file") != org) fail("wrong number of org_files for " + model_file);
	if (org.empty()) fail("no org_file for " + model_file);
	return org;
}

// ---------------------------------------------------------------------------------------------
// one array model input file
// ---------------------------------------------------------------------------------------------

void process_array_file(const Info& info, const string& model_file, const vector<size_t>& rows, ArrayCache& cache)
{
	string org_file = unique_org_file(info, rows, model_file);
	size_t skip = 0;
	if (!info.get(rows[0], "head_rows").empty()) skip = (size_t)cell_number(info.get(rows[0], "head_rows"), "head_rows");
	Array org;
	string header;
	load_array(org_file, skip, org, &header);

	if (info.has("mlt_file"))
	{
		for (size_t r : rows)
		{
			const string& mlt_file = info.get(r, "mlt_file");
			if (mlt_file.empty()) continue;
			bool multiply = operator_is_multiply(info.get(r, "operator"), mlt_file);
			shared_ptr<const Array> m = cache.get(mlt_file);
			// one-dimensional arrays (unstructured grids) take the layout of the multiplier
			if ((m->one_dim() || org.one_dim()) && m->v.size() == org.v.size())
			{
				org.nrow = m->nrow;
				org.ncol = m->ncol;
			}
			if (org.nrow != m->nrow || org.ncol != m->ncol)
				fail("shape of org file " + org_file + ":(" + to_string(org.nrow) + ", " + to_string(org.ncol) + ") differs from mlt file " + mlt_file + ":(" +
					 to_string(m->nrow) + ", " + to_string(m->ncol) + ")");
			const double* mv = m->v.data();
			double* ov = org.v.data();
			size_t n = org.v.size();
			if (multiply)
				for (size_t i = 0; i < n; ++i) ov[i] *= mv[i];
			else
				for (size_t i = 0; i < n; ++i) ov[i] += mv[i];
		}

		auto distinct = [&](const char* col) {
			vector<string> out;
			for (size_t r : rows)
			{
				const string& c = info.get(r, col);
				if (!c.empty() && find(out.begin(), out.end(), c) == out.end()) out.push_back(c);
			}
			return out;
		};
		auto same_size = [&](const Array& a, const string& path) {
			if (a.v.size() != org.v.size()) fail("'" + path + "' does not have the same number of values as '" + org_file + "'");
		};

		// bounds are only enforced where the zone array is non-zero
		vector<char> active;
		vector<string> zone_files = distinct("zone_file");
		if (!zone_files.empty())
		{
			active.assign(org.v.size(), 0);
			for (auto& zf : zone_files)
			{
				shared_ptr<const Array> z = cache.get(zf);
				same_size(*z, zf);
				for (size_t i = 0; i < active.size(); ++i)
					if (z->v[i] != 0) active[i] = 1;
			}
		}
		auto in_zone = [&](size_t i) { return active.empty() || active[i]; };

		// which == 0: upper bound, which == 1: lower bound. A bound array file wins over a scalar.
		for (int which = 0; which < 2; ++which)
		{
			const char* file_col = which == 0 ? "upper_bound_file" : "lower_bound_file";
			const char* val_col = which == 0 ? "upper_bound" : "lower_bound";
			const char* label = which == 0 ? "upper" : "lower";
			vector<string> files = distinct(file_col);
			if (!files.empty())
			{
				if (files.size() > 1) fail(string("different ") + file_col + "s for " + org_file);
				shared_ptr<const Array> b = cache.get(files[0]);
				same_size(*b, files[0]);
				for (size_t i = 0; i < org.v.size(); ++i)
					if (in_zone(i) && (which == 0 ? org.v[i] > b->v[i] : org.v[i] < b->v[i])) org.v[i] = b->v[i];
				continue;
			}
			vector<double> vals;
			for (auto& c : distinct(val_col))
			{
				double v = cell_number(c, val_col);
				if (!std::isnan(v) && find(vals.begin(), vals.end(), v) == vals.end()) vals.push_back(v);
			}
			if (vals.empty()) continue;
			if (vals.size() > 1) fail(string("different ") + label + " bound values for " + org_file);
			double b = vals[0];
			for (size_t i = 0; i < org.v.size(); ++i)
				if (in_zone(i) && (which == 0 ? org.v[i] > b : org.v[i] < b)) org.v[i] = b;
		}
	}

	string fmt_text = info.has("fmt") ? info.get(rows[0], "fmt") : string();
	if (fmt_text.empty()) fmt_text = "%15.6E";
	string sep = info.has("sep") ? info.get(rows[0], "sep") : string();
	if (sep.empty()) sep = " ";
	Fmt fmt = parse_fmt(fmt_text);
	bool one_per_value = fmt.specs.size() == 1;
	if (!one_per_value && fmt.specs.size() != org.ncol)
		fail("format for '" + model_file + "' has " + to_string(fmt.specs.size()) + " conversions but the array has " + to_string(org.ncol) + " columns");

	string out;
	out.reserve(header.size() + org.v.size() * (size_t)(max(fmt.specs[0].width, 14) + 2) + org.nrow + 64);
	out += header;
	const FmtSpec& s0 = fmt.specs[0];
	for (size_t r = 0; r < org.nrow; ++r)
	{
		const double* row = org.v.data() + r * org.ncol;
		if (one_per_value)
		{
			for (size_t c = 0; c < org.ncol; ++c)
			{
				if (c) out += sep;
				out += s0.before;
				format_number(out, s0, row[c], false);
				out += fmt.tail;
			}
		}
		else
		{
			for (size_t c = 0; c < org.ncol; ++c)
			{
				out += fmt.specs[c].before;
				format_number(out, fmt.specs[c], row[c], false);
			}
			out += fmt.tail;
		}
		out.push_back('\n');
	}
	write_file(model_file, out);
}

// ---------------------------------------------------------------------------------------------
// one list model input file
// ---------------------------------------------------------------------------------------------

// How the values in one index column are compared between the org file and a multiplier file.
// Numbers compare by value (so "3", "3.0" and " 3" are the same index); anything else by its text.
struct ColumnKeys
{
	bool all_numeric = true; // every entry reads as a number (an empty entry counts as a missing number)
	bool all_whole = true;   // ... and every one of them is a whole number
	vector<char> is_num;
	vector<double> num;
};

ColumnKeys classify_column(const vector<string_view>& cells, size_t nrow, size_t ncol, size_t col)
{
	ColumnKeys k;
	k.is_num.resize(nrow);
	k.num.resize(nrow);
	for (size_t r = 0; r < nrow; ++r)
	{
		string_view t = cells[r * ncol + col];
		double v;
		if (trim(t).empty()) { k.is_num[r] = 1; k.num[r] = nan(""); k.all_whole = false; }
		else if (whole_token_number(t, v)) { k.is_num[r] = 1; k.num[r] = v; if (!is_whole(v)) k.all_whole = false; }
		else { k.is_num[r] = 0; k.all_numeric = false; }
	}
	return k;
}

void append_key_part(string& key, const ColumnKeys& k, size_t r, string_view text, long add)
{
	if (k.is_num[r])
	{
		double v = k.num[r];
		// multiplier files store whole-number index values zero-based; text and real values are left alone
		if (add != 0 && is_whole(v) && (!k.all_numeric || k.all_whole)) v += add;
		key.push_back('n');
		if (is_whole(v)) append_int(key, (long long)v);
		else append_py_repr(key, v);
	}
	else
	{
		key.push_back('s');
		key.append(text.data(), text.size());
	}
	key.push_back('\x1f');
}

struct ListMult
{
	size_t row;              // row of the info table
	vector<string> idx_cols; // index column names (positions are turned into text)
	vector<string> use_cols;
};

void process_list_file(const Info& info, const string& model_file, const vector<size_t>& rows)
{
	string org_file = unique_org_file(info, rows, model_file);
	size_t last = rows.back();

	// output format: one fixed (printf style) format for the file, otherwise "free" with a separator
	string fmt_text;
	for (size_t r : rows)
	{
		const string& f = info.get(r, "fmt");
		if (f.empty() || f == "free") continue;
		if (!fmt_text.empty() && f != fmt_text) fail("too many different format specifiers for model file: " + model_file);
		fmt_text = f;
	}
	bool free_fmt = fmt_text.empty();
	char sep = ',';
	if (free_fmt)
	{
		string s;
		for (size_t r : rows)
		{
			const string& c = info.get(r, "sep");
			if (c.empty()) continue;
			if (!s.empty() && c != s) fail("too many different sep specifiers for model file: " + model_file);
			s = c;
		}
		if (s.empty()) fail("no separator ('sep') given for model file: " + model_file);
		if (s.size() != 1) fail("separator '" + s + "' for model file " + model_file + " must be a single character");
		sep = s[0];
	}
	size_t head_rows = info.get(last, "head_rows").empty() ? 0 : (size_t)cell_number(info.get(last, "head_rows"), "head_rows");

	vector<ListMult> mults;
	for (size_t r : rows)
	{
		ListMult m;
		m.row = r;
		for (auto& it : parse_py_list(info.get(r, "index_cols"), "index_cols")) m.idx_cols.push_back(it.text);
		for (auto& it : parse_py_list(info.get(r, "use_cols"), "use_cols")) m.use_cols.push_back(it.text);
		if (m.idx_cols.empty()) fail("no index_cols for model file: " + model_file);
		mults.push_back(std::move(m));
	}
	// index columns given by name mean the table has a line of column names; by position, it does not
	bool has_names = parse_py_list(info.get(last, "index_cols"), "index_cols")[0].is_text;

	// --- read the org file; every cell is kept as the text it was written with
	string obuf = read_file(org_file);
	const char* p = obuf.data();
	const char* e = p + obuf.size();
	for (size_t i = 0; i < head_rows && p < e; ++i)
	{
		const char* nl = (const char*)memchr(p, '\n', e - p);
		p = nl ? nl + 1 : e;
	}
	string_view head_text(obuf.data(), p - obuf.data());
	deque<string> arena;
	vector<string_view> names, f, cells;
	size_t ncol = 0, nrow = 0;
	bool need_names = has_names;
	while (p < e)
	{
		next_csv_record(p, e, f, arena);
		if (record_is_blank(f)) continue;
		if (need_names) { names = f; ncol = f.size(); need_names = false; continue; }
		if (ncol == 0) ncol = f.size();
		if (f.size() > ncol) fail("data line " + to_string(nrow + 1) + " of '" + org_file + "' has more entries than the first line");
		f.resize(ncol, string_view());
		cells.insert(cells.end(), f.begin(), f.end());
		++nrow;
	}
	if (ncol == 0) fail("no data in '" + org_file + "'");
	auto org_col = [&](const string& name) -> size_t {
		if (has_names)
		{
			for (size_t i = 0; i < names.size(); ++i)
				if (names[i] == name) return i;
			fail("column '" + name + "' not found in '" + org_file + "'");
		}
		double v;
		if (!whole_token_number(name, v) || v < 0 || v >= (double)ncol) fail("column '" + name + "' not found in '" + org_file + "'");
		return (size_t)v;
	};

	vector<double> val(nrow * ncol, 0.0); // values of the cells a multiplier has changed
	vector<char> changed(nrow * ncol, 0);
	map<size_t, ColumnKeys> org_keys;                                  // per index column of the org file
	map<string, unordered_map<string, vector<uint32_t>>> lookups;      // per set of index columns

	for (const ListMult& m : mults)
	{
		vector<size_t> oi, ou;
		string idx_tag;
		for (auto& n : m.idx_cols)
		{
			oi.push_back(org_col(n));
			idx_tag += n;
			idx_tag.push_back('\x1f');
			if (!org_keys.count(oi.back())) org_keys.emplace(oi.back(), classify_column(cells, nrow, ncol, oi.back()));
		}
		for (auto& n : m.use_cols) ou.push_back(org_col(n));
		const string& mlt_file = info.get(m.row, "mlt_file");
		if (mlt_file.empty()) continue; // a "direct" parameter: PEST writes the org file itself
		bool multiply = operator_is_multiply(info.get(m.row, "operator"), mlt_file);
		long add = 0;
		{
			string zb = info.get(m.row, "zero_based");
			for (auto& ch : zb) ch = (char)tolower((unsigned char)ch);
			if (zb == "false" || zb == "0") add = 1;
		}

		auto lit = lookups.find(idx_tag);
		if (lit == lookups.end())
		{
			unordered_map<string, vector<uint32_t>> lk;
			lk.reserve(nrow * 2);
			string key;
			for (size_t r = 0; r < nrow; ++r)
			{
				key.clear();
				for (size_t c : oi) append_key_part(key, org_keys.at(c), r, cells[r * ncol + c], 0);
				lk[key].push_back((uint32_t)r);
			}
			lit = lookups.emplace(idx_tag, std::move(lk)).first;
		}
		const auto& lookup = lit->second;

		// --- the multiplier file: a csv with a line of column names
		string mbuf = read_file(mlt_file);
		const char* q = mbuf.data();
		const char* qe = q + mbuf.size();
		deque<string> marena;
		vector<string_view> mnames, mcells;
		next_csv_record(q, qe, mnames, marena);
		size_t mncol = mnames.size(), mnrow = 0;
		auto mlt_col = [&](const string& name) -> size_t {
			for (size_t i = 0; i < mnames.size(); ++i)
				if (mnames[i] == name) return i;
			fail("column '" + name + "' not found in multiplier file '" + mlt_file + "'");
		};
		vector<size_t> mi, mu;
		for (auto& n : m.idx_cols) mi.push_back(mlt_col(n));
		for (auto& n : m.use_cols) mu.push_back(mlt_col(n));
		while (q < qe)
		{
			next_csv_record(q, qe, f, marena);
			if (record_is_blank(f)) continue;
			if (f.size() > mncol) fail("line " + to_string(mnrow + 2) + " of '" + mlt_file + "' has more entries than its first line");
			f.resize(mncol, string_view());
			mcells.insert(mcells.end(), f.begin(), f.end());
			++mnrow;
		}
		vector<ColumnKeys> mkeys;
		for (size_t c : mi) mkeys.push_back(classify_column(mcells, mnrow, mncol, c));

		size_t nmatch = 0;
		unordered_map<string, char> seen;
		seen.reserve(mnrow * 2);
		string key;
		for (size_t r = 0; r < mnrow; ++r)
		{
			key.clear();
			for (size_t c = 0; c < mi.size(); ++c) append_key_part(key, mkeys[c], r, mcells[r * mncol + mi[c]], add);
			auto it = lookup.find(key);
			if (it == lookup.end()) continue;
			if (!seen.emplace(key, 1).second) fail("multiplier file '" + mlt_file + "' has the same index values on more than one line (line " + to_string(r + 2) + ")");
			++nmatch;
			for (size_t u = 0; u < mu.size(); ++u)
			{
				string_view mt = mcells[r * mncol + mu[u]];
				double mv = nan("");
				if (!trim(mt).empty() && !whole_token_number(mt, mv)) fail("cannot read the value '" + string(mt) + "' on line " + to_string(r + 2) + " of '" + mlt_file + "'");
				for (uint32_t orow : it->second)
				{
					size_t ci = (size_t)orow * ncol + ou[u];
					if (!changed[ci])
					{
						string_view ot = cells[ci];
						if (trim(ot).empty()) val[ci] = nan("");
						else if (!whole_token_number(ot, val[ci]))
							fail("cannot apply '" + mlt_file + "': the entry '" + string(ot) + "' in column '" + m.use_cols[u] + "', data line " + to_string(orow + 1) + " of '" + org_file + "' is not a number");
						changed[ci] = 1;
					}
					if (multiply) val[ci] *= mv;
					else val[ci] += mv;
				}
			}
		}
		const string& chk = info.get(m.row, "chkpar");
		if (!chk.empty())
		{
			double want = cell_number(chk, "chkpar");
			if ((double)nmatch != want)
				fail("Probable miss-alignment in tpl indices and original file: " + to_string(nmatch) + " index values of '" + mlt_file + "' were found in '" + org_file +
					 "', expected " + to_string((long long)want));
		}
	}

	// --- ultimate bounds, per column: the largest upper and smallest lower bound given for it.
	// As in pyemu, only cells that a multiplier has changed are checked.
	for (int which = 0; which < 2; ++which)
	{
		const char* col = which == 0 ? "upper_bound" : "lower_bound";
		if (!info.has(col)) continue;
		map<size_t, double> bound;
		for (const ListMult& m : mults)
		{
			vector<PyItem> b = parse_py_list(info.get(m.row, col), col);
			for (size_t u = 0; u < m.use_cols.size() && u < b.size(); ++u)
			{
				if (b[u].is_none || b[u].is_text || std::isnan(b[u].num)) continue;
				size_t c = org_col(m.use_cols[u]);
				auto it = bound.find(c);
				if (it == bound.end()) bound[c] = b[u].num;
				else it->second = which == 0 ? max(it->second, b[u].num) : min(it->second, b[u].num);
			}
		}
		for (auto& cb : bound)
			for (size_t r = 0; r < nrow; ++r)
			{
				size_t ci = r * ncol + cb.first;
				if (changed[ci] && (which == 0 ? val[ci] > cb.second : val[ci] < cb.second)) val[ci] = cb.second;
			}
	}

	// --- write
	string out;
	out.reserve(obuf.size() + nrow * 16);
	out.append(head_text.data(), head_text.size());
	auto index_cell = [&](string& dst, size_t r, const ColumnKeys& k) {
		// pyemu converts index columns to numbers, so they are written back as numbers
		if (!k.is_num[r]) return false;
		double v = k.num[r];
		if (std::isnan(v)) return true; // written as an empty entry
		if (is_whole(v) && (!k.all_numeric || k.all_whole)) append_int(dst, (long long)v);
		else append_py_repr(dst, v);
		return true;
	};
	if (free_fmt)
	{
		if (has_names)
		{
			for (size_t c = 0; c < ncol; ++c)
			{
				if (c) out.push_back(sep);
				append_csv_field(out, names[c], sep);
			}
			out.push_back('\n');
		}
		vector<const ColumnKeys*> keyed(ncol, nullptr);
		for (auto& kv : org_keys) keyed[kv.first] = &kv.second;
		for (size_t r = 0; r < nrow; ++r)
		{
			for (size_t c = 0; c < ncol; ++c)
			{
				if (c) out.push_back(sep);
				size_t ci = r * ncol + c;
				if (changed[ci])
				{
					if (!std::isnan(val[ci])) append_py_repr(out, val[ci]);
				}
				else if (keyed[c] && index_cell(out, r, *keyed[c])) {}
				else append_csv_field(out, cells[ci], sep);
			}
			out.push_back('\n');
		}
	}
	else
	{
		// fixed format: each column is written as numbers if every entry in it is a number
		Fmt fmt = parse_fmt(fmt_text);
		bool one_per_value = fmt.specs.size() == 1;
		if (!one_per_value && fmt.specs.size() != ncol)
			fail("format for '" + model_file + "' has " + to_string(fmt.specs.size()) + " conversions but the table has " + to_string(ncol) + " columns");
		vector<char> col_numeric(ncol, 1), col_int(ncol, 1);
		vector<double> num(nrow * ncol, 0.0);
		for (size_t c = 0; c < ncol; ++c)
			for (size_t r = 0; r < nrow; ++r)
			{
				size_t ci = r * ncol + c;
				if (changed[ci]) { num[ci] = val[ci]; col_int[c] = 0; continue; }
				string_view t = trim(cells[ci]);
				if (t.empty()) { num[ci] = nan(""); col_int[c] = 0; continue; }
				if (!whole_token_number(t, num[ci])) { col_numeric[c] = 0; continue; }
				long long iv;
				auto pr = from_chars(t.data(), t.data() + t.size(), iv);
				if (pr.ec != errc() || pr.ptr != t.data() + t.size()) col_int[c] = 0;
			}
		for (size_t r = 0; r < nrow; ++r)
		{
			for (size_t c = 0; c < ncol; ++c)
			{
				const FmtSpec& sp = one_per_value ? fmt.specs[0] : fmt.specs[c];
				if (one_per_value && c) out.push_back(' ');
				out += sp.before;
				size_t ci = r * ncol + c;
				bool is_number = changed[ci] || (col_numeric[c] ? true : whole_token_number(cells[ci], num[ci]));
				if (is_number) format_number(out, sp, num[ci], col_numeric[c] && col_int[c]);
				else format_text(out, sp, cells[ci]);
				if (one_per_value) out += fmt.tail;
			}
			if (!one_per_value) out += fmt.tail;
			out.push_back('\n');
		}
	}
	write_file(model_file, out);
}

// ---------------------------------------------------------------------------------------------
// pilot points: multiplier arrays from pilot point values and a text file of kriging factors
// (the same job as pyemu.geostats.fac2real with a PEST-style text factor file)
// ---------------------------------------------------------------------------------------------

// pilot point file: name x y zone value, separated by blanks
void read_pp_file(const string& path, vector<string>& names, vector<double>& values)
{
	string buf = read_file(path);
	const char* p = buf.data();
	const char* e = p + buf.size();
	size_t line_no = 0;
	while (p < e)
	{
		const char* nl = (const char*)memchr(p, '\n', e - p);
		const char* le = nl ? nl : e;
		++line_no;
		vector<string_view> tok;
		const char* q = p;
		while (q < le)
		{
			while (q < le && (*q == ' ' || *q == '\t' || *q == '\r')) ++q;
			const char* s = q;
			while (q < le && !(*q == ' ' || *q == '\t' || *q == '\r')) ++q;
			if (q > s) tok.emplace_back(s, q - s);
		}
		p = nl ? nl + 1 : e;
		if (tok.empty()) continue;
		double v;
		if (tok.size() < 5 || !whole_token_number(tok[4], v)) fail("cannot read line " + to_string(line_no) + " of pilot point file '" + path + "'");
		string n(tok[0]);
		for (auto& ch : n) ch = (char)tolower((unsigned char)ch);
		names.push_back(n);
		values.push_back(v);
	}
}

void write_npy(const string& path, const Array& a)
{
	string hdr = "{'descr': '<f8', 'fortran_order': False, 'shape': (" + to_string(a.nrow) + ", " + to_string(a.ncol) + "), }";
	while ((10 + hdr.size() + 1) % 64 != 0) hdr.push_back(' ');
	hdr.push_back('\n');
	string out("\x93NUMPY\x01\x00", 8);
	out.push_back((char)(hdr.size() & 0xff));
	out.push_back((char)((hdr.size() >> 8) & 0xff));
	out += hdr;
	out.append((const char*)a.v.data(), a.v.size() * sizeof(double));
	write_file(path, out);
}

struct PilotPointTask
{
	string pp_file, fac_file, out_file;
	double fill = 1.0e30, lower = -1.0e30, upper = 1.0e30;
	bool operator<(const PilotPointTask& o) const
	{
		return tie(out_file, pp_file, fac_file, fill, lower, upper) < tie(o.out_file, o.pp_file, o.fac_file, o.fill, o.lower, o.upper);
	}
};

// One multiplier array: reads the pilot point values, then goes through the factor file once,
// computing each array cell as its entry is read (nothing from the factor file is kept).
void run_pilot_point_task(const PilotPointTask& t)
{
	vector<string> names;
	vector<double> values;
	read_pp_file(t.pp_file, names, values);
	unordered_map<string, double> by_name;
	for (size_t i = 0; i < names.size(); ++i) by_name[names[i]] = values[i];

	const string& path = t.fac_file;
	string buf = read_file(path);
	if (memchr(buf.data(), 0, min<size_t>(buf.size(), 4096)))
		fail("'" + path + "' is not a text factor file (binary factor files written by pypestutils are not supported; the caller must make those multiplier arrays and run this program with --skip_pilot_points)");
	const char* p = buf.data();
	const char* e = p + buf.size();
	auto next_line = [&]() -> string_view {
		const char* s = p;
		const char* nl = (const char*)memchr(p, '\n', e - p);
		p = nl ? nl + 1 : e;
		return trim(string_view(s, (nl ? nl : e) - s));
	};
	auto skip_blank = [&]() {
		while (p < e && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) ++p;
	};
	auto next_int = [&](const char* what) -> long long {
		skip_blank();
		long long v;
		auto r = from_chars(p, e, v);
		if (r.ec != errc()) fail("cannot read " + string(what) + " in factor file '" + path + "'");
		p = r.ptr;
		return v;
	};
	auto next_real = [&](const char* what) -> double {
		skip_blank();
		double v;
		const char* r = parse_double(p, e, v);
		if (!r) fail("cannot read " + string(what) + " in factor file '" + path + "'");
		p = r;
		return v;
	};
	next_line(); // name of the pilot point file the factors were made from
	next_line(); // name of the zone file
	long long ncol = next_int("the number of columns"), nrow = next_int("the number of rows");
	long long npp = next_int("the number of pilot points");
	if (ncol < 1 || nrow < 1 || npp < 1) fail("bad array size or pilot point count in factor file '" + path + "'");
	next_line(); // rest of the pilot point count line
	if ((size_t)npp != names.size())
		fail("pilot point file '" + t.pp_file + "' has " + to_string(names.size()) + " points but factor file '" + path + "' refers to " + to_string(npp));
	// pilot point values in the order the factor file numbers them, matched by name
	vector<double> val((size_t)npp), logval((size_t)npp);
	for (long long i = 0; i < npp; ++i)
	{
		string n(next_line());
		for (auto& ch : n) ch = (char)tolower((unsigned char)ch);
		auto it = by_name.find(n);
		if (it == by_name.end()) fail("pilot point '" + n + "' of factor file '" + path + "' is not in pilot point file '" + t.pp_file + "'");
		val[i] = it->second;
		logval[i] = std::log10(it->second);
	}

	Array a;
	a.nrow = (size_t)nrow;
	a.ncol = (size_t)ncol;
	a.v.assign(a.nrow * a.ncol, t.fill);
	// each entry: cell number, transform flag, number of factors, a value that is not used, then
	// (pilot point number, weight) pairs; an entry may run over several lines
	while (true)
	{
		skip_blank();
		if (p >= e) break;
		long long inode = next_int("a cell number"), itrans = next_int("a transform flag"), nfac = next_int("a factor count");
		if (inode < 1 || (size_t)inode > a.v.size()) fail("cell number " + to_string(inode) + " is outside the array in factor file '" + path + "'");
		next_real("an entry");
		const vector<double>& src = itrans != 0 ? logval : val;
		double sum = 0.0;
		for (long long k = 0; k < nfac; ++k)
		{
			long long ipp = next_int("a pilot point number");
			if (ipp < 1 || ipp > npp) fail("pilot point number " + to_string(ipp) + " is out of range for cell " + to_string(inode) + " in factor file '" + path + "'");
			sum += src[ipp - 1] * next_real("a weight");
		}
		a.v[inode - 1] = itrans != 0 ? std::pow(10.0, sum) : sum;
	}
	for (double& x : a.v)
	{
		if (x < t.lower) x = t.lower;
		if (x > t.upper) x = t.upper;
	}
	if (ends_with(t.out_file, ".npy")) { write_npy(t.out_file, a); return; }
	// text, the way pyemu writes it: "%15.6E" with nothing between the values
	Fmt fmt = parse_fmt("%15.6E");
	string out;
	out.reserve(a.v.size() * 15 + a.nrow + 16);
	for (size_t r = 0; r < a.nrow; ++r)
	{
		for (size_t c = 0; c < a.ncol; ++c) format_number(out, fmt.specs[0], a.v[r * a.ncol + c], false);
		out.push_back('\n');
	}
	write_file(t.out_file, out);
}

// Runs fn(i) for i in [0, n) on up to `nthreads` threads; error messages are collected.
template <class F> void run_parallel(size_t n, unsigned nthreads, vector<string>& errors, F fn)
{
	atomic<size_t> next(0);
	mutex err_mtx;
	auto worker = [&]() {
		size_t i;
		while ((i = next++) < n)
		{
			try { fn(i); }
			catch (const exception& ex)
			{
				lock_guard<mutex> g(err_mtx);
				errors.push_back(ex.what());
			}
		}
	};
	unsigned nt = (unsigned)min<size_t>(nthreads, max<size_t>(n, 1));
	vector<thread> threads;
	for (unsigned t = 1; t < nt; ++t) threads.emplace_back(worker);
	worker();
	for (auto& t : threads) t.join();
}

// ---------------------------------------------------------------------------------------------
// driver
// ---------------------------------------------------------------------------------------------

struct Job
{
	bool is_list;
	string model_file;
	vector<size_t> rows;
};

void usage()
{
	cout << "usage: pestpp-appl [mult2model_info.csv] [--apply_order <value>] [--threads <n>] [--ignore_functions] [--skip_pilot_points]\n"
		 << "\n"
		 << "  applies the multiplier files listed in the table written by pyemu's PstFrom to the\n"
		 << "  model input files (a compiled equivalent of pyemu.helpers.apply_list_and_array_pars).\n"
		 << "\n"
		 << "  --apply_order <value>  only process rows with this 'apply_order' value\n"
		 << "  --threads <n>          number of threads to use (default: 1). this program\n"
		 << "                         runs inside each model run, often with many agents on one\n"
		 << "                         machine, so only raise it when the machine has cores to spare\n"
		 << "  -v, --version          print the version number and stop\n"
		 << "  --ignore_functions     process the whole table even though it has entries in the\n"
		 << "                         'pre_apply_function'/'post_apply_function' columns (Python\n"
		 << "                         code this program cannot run)\n"
		 << "  --skip_pilot_points    do not make multiplier arrays from pilot point files; the\n"
		 << "                         caller has already written them\n";
}

int run(int argc, char* argv[])
{
	string info_file = "mult2model_info.csv";
	bool have_order = false, ignore_functions = false, skip_pp = false;
	double only_order = 0;
	unsigned nthreads = 1;
	for (int i = 1; i < argc; ++i)
	{
		string a = argv[i];
		auto value = [&]() -> string {
			if (i + 1 >= argc) fail("missing value after '" + a + "'");
			return argv[++i];
		};
		if (a == "-h" || a == "--help") { usage(); return 0; }
		else if (a == "-v" || a == "--version") { cout << PESTPP_VERSION << endl; return 0; }
		else if (a == "--apply_order") { only_order = cell_number(value(), "--apply_order"); have_order = true; }
		else if (a == "--threads") { nthreads = (unsigned)max(1.0, cell_number(value(), "--threads")); }
		else if (a == "--ignore_functions") ignore_functions = true;
		else if (a == "--skip_pilot_points") skip_pp = true;
		else if (!a.empty() && a[0] == '-') fail("unrecognized argument '" + a + "'");
		else info_file = a;
	}

	auto t_start = chrono::steady_clock::now();
	Info info = read_info(info_file);

	auto order_of = [&](size_t r) {
		const string& c = info.get(r, "apply_order");
		return c.empty() ? 999.0 : cell_number(c, "apply_order");
	};
	set<double> orders;
	for (size_t r = 0; r < info.rows.size(); ++r) orders.insert(order_of(r));
	if (have_order)
	{
		if (!orders.count(only_order)) fail("no rows with apply_order " + to_string(only_order) + " in '" + info_file + "'");
		orders = {only_order};
	}
	else if (!ignore_functions)
	{
		for (const char* col : {"pre_apply_function", "post_apply_function"})
			for (size_t r = 0; r < info.rows.size(); ++r)
				if (!info.get(r, col).empty())
					fail(string("'") + info_file + "' has an entry in the '" + col + "' column ('" + info.get(r, col) +
						 "'). That is Python code this program cannot run: run it from the caller and use --apply_order for each apply_order value in turn, or pass --ignore_functions");
	}

	size_t n_array = 0, n_list = 0, n_pp = 0;
	vector<string> errors;
	for (double order : orders)
	{
		// one job per model input file, in the order the files first appear; list files first
		vector<Job> jobs;
		for (int pass = 0; pass < 2; ++pass)
		{
			bool want_list = pass == 0;
			unordered_map<string, size_t> where;
			for (size_t r = 0; r < info.rows.size(); ++r)
			{
				if (order_of(r) != order) continue;
				bool is_list = !info.get(r, "index_cols").empty();
				if (is_list != want_list) continue;
				const string& mf = info.get(r, "model_file");
				auto it = where.find(mf);
				if (it == where.end())
				{
					where[mf] = jobs.size();
					jobs.push_back(Job{is_list, mf, {r}});
				}
				else
					jobs[it->second].rows.push_back(r);
			}
		}

		// array files used by more than one model input file are read once
		map<string, int> uses;
		for (const Job& j : jobs)
		{
			if (j.is_list) continue;
			set<string> mine;
			for (size_t r : j.rows)
				for (const char* col : {"mlt_file", "zone_file", "upper_bound_file", "lower_bound_file"})
					if (!info.get(r, col).empty()) mine.insert(info.get(r, col));
			for (auto& s : mine) ++uses[s];
		}
		set<string> shared;
		for (auto& u : uses)
			if (u.second > 1) shared.insert(u.first);
		ArrayCache cache;
		cache.set_shared(shared);

		// pilot points first: they produce multiplier arrays that the array files below read
		if (!skip_pp && info.has("pp_file"))
		{
			// one task per multiplier file, so no two threads ever write the same file; rows that
			// name the same file must agree on everything that goes into it
			map<string, PilotPointTask> task_by_file;
			for (size_t r = 0; r < info.rows.size(); ++r)
			{
				if (order_of(r) != order || info.get(r, "pp_file").empty() || !info.get(r, "index_cols").empty()) continue;
				PilotPointTask t;
				t.pp_file = info.get(r, "pp_file");
				t.fac_file = info.get(r, "fac_file");
				t.out_file = info.get(r, "mlt_file");
				if (t.fac_file.empty() || t.out_file.empty()) { errors.push_back("row with pp_file '" + t.pp_file + "' needs both fac_file and mlt_file"); continue; }
				if (!info.get(r, "pp_fill_value").empty()) t.fill = cell_number(info.get(r, "pp_fill_value"), "pp_fill_value");
				if (!info.get(r, "pp_lower_limit").empty()) t.lower = cell_number(info.get(r, "pp_lower_limit"), "pp_lower_limit");
				if (!info.get(r, "pp_upper_limit").empty()) t.upper = cell_number(info.get(r, "pp_upper_limit"), "pp_upper_limit");
				auto it = task_by_file.find(t.out_file);
				if (it == task_by_file.end()) task_by_file[t.out_file] = t;
				else if (it->second < t || t < it->second)
					errors.push_back("rows write the pilot point multiplier file '" + t.out_file + "' with different pp_file, fac_file, pp_fill_value, pp_lower_limit or pp_upper_limit entries");
			}
			if (!errors.empty()) break;
			vector<PilotPointTask> tasks;
			for (auto& kv : task_by_file) tasks.push_back(kv.second);
			run_parallel(tasks.size(), nthreads, errors, [&](size_t i) {
				try { run_pilot_point_task(tasks[i]); }
				catch (const exception& ex) { fail("pilot point multiplier file '" + tasks[i].out_file + "': " + ex.what()); }
			});
			n_pp += tasks.size();
			if (!errors.empty()) break;
		}

		atomic<size_t> next(0);
		mutex err_mtx;
		auto worker = [&]() {
			size_t j;
			while ((j = next++) < jobs.size())
			{
				try
				{
					if (jobs[j].is_list) process_list_file(info, jobs[j].model_file, jobs[j].rows);
					else process_array_file(info, jobs[j].model_file, jobs[j].rows, cache);
				}
				catch (const exception& ex)
				{
					lock_guard<mutex> g(err_mtx);
					errors.push_back("model file '" + jobs[j].model_file + "': " + ex.what());
				}
			}
		};
		unsigned nt = (unsigned)min<size_t>(nthreads, max<size_t>(jobs.size(), 1));
		vector<thread> threads;
		for (unsigned t = 1; t < nt; ++t) threads.emplace_back(worker);
		worker();
		for (auto& t : threads) t.join();
		for (const Job& j : jobs) (j.is_list ? n_list : n_array)++;
		if (!errors.empty()) break;
	}

	if (!errors.empty())
	{
		sort(errors.begin(), errors.end());
		for (auto& m : errors) cerr << "pestpp-appl error: " << m << "\n";
		return 1;
	}
	double secs = chrono::duration<double>(chrono::steady_clock::now() - t_start).count();
	cout << "pestpp-appl: wrote " << n_pp << " pilot point multiplier array(s), " << n_list << " list file(s) and " << n_array << " array file(s) in " << secs << " seconds using up to " << nthreads << " thread(s)" << endl;
	return 0;
}

} // namespace

int main(int argc, char* argv[])
{
	try
	{
		return run(argc, argv);
	}
	catch (const exception& ex)
	{
		cerr << "pestpp-appl error: " << ex.what() << endl;
		return 1;
	}
}
