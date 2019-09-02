#ifndef __TATOOINE_BOOSTRESIZER__
#define __TATOOINE_BOOSTRESIZER__

// #include <armadillo>
#include <boost/numeric/odeint.hpp>
#include "../../vecmat.h"

namespace boost::numeric::odeint {

// // vec
// template <>
// struct is_resizeable<arma::vec> {
//   typedef boost::true_type type;
//   static const bool        value = type::value;
// };

// template <>
// struct same_size_impl<arma::vec, arma::vec> {
//   static bool same_size(const arma::vec& x, const arma::vec& y) {
//     return x.size() == y.size();
//   }
// };
//
// template <>
// struct resize_impl<arma::vec, arma::vec> {
//   static void resize(arma::vec& v1, const arma::vec& v2) {
//     v1.set_size(v2.size());
//   }
// };
//
// // fvec
// template <>
// struct is_resizeable<arma::fvec> {
//   typedef boost::true_type type;
//   static const bool        value = type::value;
// };
//
// template <>
// struct same_size_impl<arma::fvec, arma::fvec> {
//   static bool same_size(const arma::fvec& x, const arma::fvec& y) {
//     return x.size() == y.size();
//   }
// };
//
// template <>
// struct resize_impl<arma::fvec, arma::fvec> {
//   static void resize(arma::fvec& v1, const arma::fvec& v2) {
//     v1.set_size(v2.size());
//   }
// };
//
// // cx_vec
// template <>
// struct is_resizeable<arma::cx_vec> {
//   typedef boost::true_type type;
//   static const bool        value = type::value;
// };
//
// template <>
// struct same_size_impl<arma::cx_vec, arma::cx_vec> {
//   static bool same_size(const arma::cx_vec& x, const arma::cx_vec& y) {
//     return x.size() == y.size();
//   }
// };
//
// template <>
// struct resize_impl<arma::cx_vec, arma::cx_vec> {
//   static void resize(arma::cx_vec& v1, const arma::cx_vec& v2) {
//     v1.set_size(v2.size());
//   }
// };
//
// // cx_fvec
// template <>
// struct is_resizeable<arma::cx_fvec> {
//   typedef boost::true_type type;
//   static const bool        value = type::value;
// };
//
// template <>
// struct same_size_impl<arma::cx_fvec, arma::cx_fvec> {
//   static bool same_size(const arma::cx_fvec& x, const arma::cx_fvec& y) {
//     return x.size() == y.size();
//   }
// };
//
// template <>
// struct resize_impl<arma::cx_fvec, arma::cx_fvec> {
//   static void resize(arma::cx_fvec& v1, const arma::cx_fvec& v2) {
//     v1.set_size(v2.size());
//   }
// };
//
// // ivec
// template <>
// struct is_resizeable<arma::ivec> {
//   typedef boost::true_type type;
//   static const bool        value = type::value;
// };
//
// template <>
// struct same_size_impl<arma::ivec, arma::ivec> {
//   static bool same_size(const arma::ivec& x, const arma::ivec& y) {
//     return x.size() == y.size();
//   }
// };
//
// template <>
// struct resize_impl<arma::ivec, arma::ivec> {
//   static void resize(arma::ivec& v1, const arma::ivec& v2) {
//     v1.set_size(v2.size());
//   }
// };
//
// // uvec
// template <>
// struct is_resizeable<arma::uvec> {
//   typedef boost::true_type type;
//   static const bool        value = type::value;
// };
//
// template <>
// struct same_size_impl<arma::uvec, arma::uvec> {
//   static bool same_size(const arma::uvec& x, const arma::uvec& y) {
//     return x.size() == y.size();
//   }
// };
//
// template <>
// struct resize_impl<arma::uvec, arma::uvec> {
//   static void resize(arma::uvec& v1, const arma::uvec& v2) {
//     v1.set_size(v2.size());
//   }
// };

}  // namespace boost::numeric::odeint

#endif
