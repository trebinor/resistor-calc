//! A resistor value optimiser for circuit design.
//!
//! When provided with a set of constraints and relations for a series of resistors R1, R2, ..., it
//! can present sets of values from standard series in order of increasing inaccuracy.
//!
//! # Example
//! Given the following resistor network:
//!
//! ![diagram](https://i.imgur.com/GoZKJoL.png)
//!
//! Where VADJ must remain at 0.8v, as R2 varies from no to full resistance,
//! VOUT varies from 6v to 12v
//!
//! We can then describe the problem via the following constraints, plus a few extra bounds to
//! eliminate either very small, or very large values, both of which may cause current issues.
//! ```rust no_run
//! extern crate resistor_calc;
//!
//! use resistor_calc::*;
//!
//! fn main() {
//!     let rcalc = RCalc::new(vec![&E24, &E6, &E24]);
//!
//!     println!("Number of combinations: {}", rcalc.combinations());
//!
//!     let res = rcalc
//!         .calc(
//!             ROpBuilder::new()
//!                 .bound("R1+R2+R3 <= 1e6")
//!                 .bound("R1+R2+R3 >= 1e4")
//!                 .bound("0.8 * (1 + R1/R3) ~ 6.0")
//!                 .bound("0.8 * (1 + (R1+R2)/R3) ~ 12.0")
//!                 .finish(),
//!         )
//!         .expect("Error: No values satisfy requirements");
//!
//!     res.print_best();
//! }
//! ```
//! Running this example produces the results:
//! ```text
//! Number of combinations: 1185408
//! Match 1:
//! Error: 0.000
//! Values: R1: 13K, R2: 15K, R3: 2K
//!
//! Match 2:
//! Error: 0.000
//! Values: R1: 130K, R2: 150K, R3: 20K
//!```

extern crate itertools;
#[macro_use]
extern crate lazy_static;

use itertools::Itertools;

use std::fmt;

#[cfg(feature = "expr_builder")]
mod expr_builder;

#[cfg(feature = "expr_builder")]
pub use expr_builder::ROpBuilder;

const POWERS: &[f64] = &[1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6];

lazy_static! {
    /// RSeries constant for the E3 standard series
    pub static ref E3: RSeries = RSeries::new(&[1.0, 2.2, 4.7]);
    /// RSeries constant for the E6 standard series
    pub static ref E6: RSeries = RSeries::extend(&E3, &[1.5, 3.3, 6.8]);
    /// RSeries constant for the E12 standard series
    pub static ref E12: RSeries = RSeries::extend(&E6, &[1.2, 1.8, 2.7, 3.9, 5.6, 8.2]);
    /// RSeries constant for the E24 standard series
    pub static ref E24: RSeries = RSeries::extend(
        &E12,
        &[1.1, 1.3, 1.6, 2.0, 2.4, 3.0, 3.6, 4.3, 5.1, 6.2, 7.5, 9.1],
    );
    /// RSeries constant for the E48 standard series
    pub static ref E48: RSeries = RSeries::new(&[
        1.00, 1.05, 1.10, 1.15, 1.21, 1.27, 1.33, 1.40, 1.47, 1.54, 1.62, 1.69, 1.78, 1.87, 1.96, 2.05,
        2.15, 2.26, 2.37, 2.49, 2.61, 2.74, 2.87, 3.01, 3.16, 3.32, 3.48, 3.65, 3.83, 4.02, 4.22, 4.42,
        4.64, 4.87, 5.11, 5.36, 5.62, 5.90, 6.19, 6.49, 6.81, 7.15, 7.50, 7.87, 8.25, 8.66, 9.09, 9.53,
    ]);
    /// RSeries constant for the E96 standard series
    pub static ref E96: RSeries = RSeries::new(&[
        1.00, 1.02, 1.05, 1.07, 1.10, 1.13, 1.15, 1.18, 1.21, 1.24, 1.27, 1.30, 1.33, 1.37, 1.40, 1.43,
        1.47, 1.50, 1.54, 1.58, 1.62, 1.65, 1.69, 1.74, 1.78, 1.82, 1.87, 1.91, 1.96, 2.00, 2.05, 2.10,
        2.15, 2.21, 2.26, 2.32, 2.37, 2.43, 2.49, 2.55, 2.61, 2.67, 2.74, 2.80, 2.87, 2.94, 3.01, 3.09,
        3.16, 3.24, 3.32, 3.40, 3.48, 3.57, 3.65, 3.74, 3.83, 3.92, 4.02, 4.12, 4.22, 4.32, 4.42, 4.53,
        4.64, 4.75, 4.87, 4.99, 5.11, 5.23, 5.36, 5.49, 5.62, 5.76, 5.90, 6.04, 6.19, 6.34, 6.49, 6.65,
        6.81, 6.98, 7.15, 7.32, 7.50, 7.68, 7.87, 8.06, 8.25, 8.45, 8.66, 8.87, 9.09, 9.31, 9.53, 9.76,
    ]);
    /// RSeries constant for the E192 standard series
    pub static ref E192: RSeries = RSeries::new(&[
        1.00, 1.01, 1.02, 1.04, 1.05, 1.06, 1.07, 1.09, 1.10, 1.11, 1.13, 1.14, 1.15, 1.17, 1.18, 1.20,
        1.21, 1.23, 1.24, 1.26, 1.27, 1.29, 1.30, 1.32, 1.33, 1.35, 1.37, 1.38, 1.40, 1.42, 1.43, 1.45,
        1.47, 1.49, 1.50, 1.52, 1.54, 1.56, 1.58, 1.60, 1.62, 1.64, 1.65, 1.67, 1.69, 1.72, 1.74, 1.76,
        1.78, 1.80, 1.82, 1.84, 1.87, 1.89, 1.91, 1.93, 1.96, 1.98, 2.00, 2.03, 2.05, 2.08, 2.10, 2.13,
        2.15, 2.18, 2.21, 2.23, 2.26, 2.29, 2.32, 2.34, 2.37, 2.40, 2.43, 2.46, 2.49, 2.52, 2.55, 2.58,
        2.61, 2.64, 2.67, 2.71, 2.74, 2.77, 2.80, 2.84, 2.87, 2.91, 2.94, 2.98, 3.01, 3.05, 3.09, 3.12,
        3.16, 3.20, 3.24, 3.28, 3.32, 3.36, 3.40, 3.44, 3.48, 3.52, 3.57, 3.61, 3.65, 3.70, 3.74, 3.79,
        3.83, 3.88, 3.92, 3.97, 4.02, 4.07, 4.12, 4.17, 4.22, 4.27, 4.32, 4.37, 4.42, 4.48, 4.53, 4.59,
        4.64, 4.70, 4.75, 4.81, 4.87, 4.93, 4.99, 5.05, 5.11, 5.17, 5.23, 5.30, 5.36, 5.42, 5.49, 5.56,
        5.62, 5.69, 5.76, 5.83, 5.90, 5.97, 6.04, 6.12, 6.19, 6.26, 6.34, 6.42, 6.49, 6.57, 6.65, 6.73,
        6.81, 6.90, 6.98, 7.06, 7.15, 7.23, 7.32, 7.41, 7.50, 7.59, 7.68, 7.77, 7.87, 7.96, 8.06, 8.16,
        8.25, 8.35, 8.45, 8.56, 8.66, 8.76, 8.87, 8.98, 9.09, 9.20, 9.31, 9.42, 9.53, 9.65, 9.76, 9.88,
    ]);
    pub static ref E24_E96: RSeries = RSeries::extend(&E24, &[
        1.00, 1.02, 1.05, 1.07, 1.10, 1.13, 1.15, 1.18, 1.21, 1.24, 1.27, 1.30, 1.33, 1.37, 1.40, 1.43,
        1.47, 1.50, 1.54, 1.58, 1.62, 1.65, 1.69, 1.74, 1.78, 1.82, 1.87, 1.91, 1.96, 2.00, 2.05, 2.10,
        2.15, 2.21, 2.26, 2.32, 2.37, 2.43, 2.49, 2.55, 2.61, 2.67, 2.74, 2.80, 2.87, 2.94, 3.01, 3.09,
        3.16, 3.24, 3.32, 3.40, 3.48, 3.57, 3.65, 3.74, 3.83, 3.92, 4.02, 4.12, 4.22, 4.32, 4.42, 4.53,
        4.64, 4.75, 4.87, 4.99, 5.11, 5.23, 5.36, 5.49, 5.62, 5.76, 5.90, 6.04, 6.19, 6.34, 6.49, 6.65,
        6.81, 6.98, 7.15, 7.32, 7.50, 7.68, 7.87, 8.06, 8.25, 8.45, 8.66, 8.87, 9.09, 9.31, 9.53, 9.76,
    ]);
}

pub(crate) fn _test_calc() -> RRes {
    let r = RCalc::e3(2);
    r.calc(ROpBuilder::new().bound("R1 + R2 ~ 500").finish())
        .unwrap()
}

/// A series of resistor values, constants are provided for standard resistor array values.
#[derive(Debug)]
pub struct RSeries {
    pub values: Box<[f64]>,
}

impl RSeries {
    /// Defines a new series of resistor values. Only the decade needs to be provided, multiples
    /// upto Val * 1M will be generated automatically.
    /// # Example
    /// ```
    ///     # use resistor_calc::*;
    ///     let piher = RSeries::new(&[1.0, 2.0, 2.2, 2.5, 4.7, 5.0]);
    /// ```
    pub fn new(series: &[f64]) -> Self {
        RSeries {
            values: series
                .iter()
                .cartesian_product(POWERS.iter())
                .map(|(val, pow)| val * pow)
                .collect::<Vec<f64>>()
                .into_boxed_slice(),
        }
    }

    fn extend(base: &RSeries, add: &[f64]) -> Self {
        RSeries {
            values: base
                .iter()
                .cloned()
                .chain(
                    add.iter()
                        .cartesian_product(POWERS.iter())
                        .map(|(val, pow)| val * pow),
                )
                .sorted_by(|a, b| a.partial_cmp(b).expect("No NaNs"))
                .iter()
                .dedup()
                .cloned()
                .collect::<Vec<f64>>()
                .into_boxed_slice(),
        }
    }

    fn iter(&self) -> impl Iterator<Item = &f64> + Clone {
        self.values.iter()
    }

    fn len(&self) -> usize {
        self.values.len()
    }
}

fn _format_rval(r: f64, unit: &str) -> String {
    let mut val = format!("{}", r);
    if val.contains('.') {
        val.replace(".", unit)
    } else {
        val.push_str(unit);
        val
    }
}

fn _print_r(r: &f64) -> String {
    if *r < 1000.0 {
        _format_rval(*r, "R")
    } else if *r < 1_000_000.0 {
        _format_rval(*r / 1000.0, "K")
    } else {
        _format_rval(*r / 1_000_000.0, "M")
    }
}

fn _print_res(r: &(u64, RSet)) {
    let &(r, ref v) = r;
    println!("Error: {:.3}\nValues: {}", (r as f64) / 1e9, v);
}

/// A binding of values to the set of resistors in a calculation.
#[derive(Debug)]
pub struct RSet(Box<[f64]>);

impl RSet {
    /// Retrieves the value of R{idx}, starting from R1, R2, ..., Rn
    /// # Examples
    /// ```
    ///     # let ret = {
    ///     #     use resistor_calc::{RCalc, ROpBuilder};
    ///     #     let r = RCalc::e3(2);
    ///     #     r.calc(ROpBuilder::new()
    ///     #         .bound("R1 + R2 ~ 500")
    ///     #         .finish()
    ///     #     ).unwrap()
    ///     # };
    ///     for (err, rset) in ret.iter() {
    ///         println!("R1 = {}", rset.r(1));
    ///         println!("R2 = {}", rset.r(2));
    ///     }
    /// ```
    pub fn r(&self, idx: usize) -> f64 {
        self.0[idx - 1]
    }

    /// Returns the sum of all the values in the set. Good for presenting overall bounds on dividers.
    pub fn sum(&self) -> f64 {
        self.0.iter().sum()
    }
}

impl fmt::Display for RSet {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let sep = if f.alternate() { "\n" } else { ", " };
        write!(
            f,
            "{}",
            self.0
                .iter()
                .enumerate()
                .map(|(i, r)| format!("R{}: {}", i + 1, _print_r(r)))
                .join(sep)
        )
    }
}

/// Stores the result of a calculation.
#[derive(Debug)]
pub struct RRes {
    res: Vec<(u64, RSet)>,
}

impl RRes {
    /// Print all combinations that share the lowest error value.
    pub fn print_best(&self) {
        let best_err = self.res[0].0;
        for (idx, res) in self
            .res
            .iter()
            .take_while(|(err, _)| *err == best_err)
            .enumerate()
        {
            println!("Match {}:", idx + 1);
            _print_res(res);
            println!();
        }
    }

    /// Provides an iterator over all results in the object. They are presented from lowest to
    /// highest error value, within a given error value different combinations may be presented in
    /// any order. The item type is `&(u64, RSet)`, where the first value is parts in a billion
    /// error (`floor(err * 1e9)`).
    pub fn iter(&self) -> impl Iterator<Item = &(u64, RSet)> {
        self.res.iter()
    }
}

/// Main calculator struct
#[derive(Debug)]
pub struct RCalc<'a> {
    rs: Vec<&'a RSeries>,
}

impl<'a> RCalc<'a> {
    /// Creates a new RCalc with the series used for the R values provided as a vec.
    /// # Examples
    /// To create a calculator that will vary over 4 resistors R1, R2, R3 and R4, where we want to
    /// draw R1 and R2 from the E24 series, R3 from the E6 series and R4 from the E12 series would
    /// be done as follows:
    /// ```
    ///     # use resistor_calc::*;
    ///     let rcal = RCalc::new(vec![&E24, &E24, &E6, &E12]);
    /// ```
    pub fn new(rs: Vec<&'a RSeries>) -> Self {
        RCalc { rs }
    }

    /// Creates a new RCalc with `count` resistors drawn from the E3 series.
    pub fn e3(count: usize) -> Self {
        Self::new(vec![&E3; count])
    }

    /// Creates a new RCalc with `count` resistors drawn from the E6 series.
    pub fn e6(count: usize) -> Self {
        Self::new(vec![&E6; count])
    }

    /// Creates a new RCalc with `count` resistors drawn from the E12 series.
    pub fn e12(count: usize) -> Self {
        Self::new(vec![&E12; count])
    }

    /// Creates a new RCalc with `count` resistors drawn from the E24 series.
    pub fn e24(count: usize) -> Self {
        Self::new(vec![&E24; count])
    }

    /// Returns the number of combinations of values that exist for the configured resistors and
    /// series. This will fairly directly map to the amount of time taken to calculate value
    /// combinations.
    pub fn combinations(&self) -> u128 {
        self.rs.iter().map(|r| r.len() as u128).product()
    }

    /// Given a testing function `f` thats maps from a set of input resistors to `Option<f64>` this
    /// will calculate the results for the resistors and series configured and return the result as
    /// an `RRes`. `f` should map combinations that are unsuitable to `None` and combinations that
    /// are suitable to `Some(err)` where `err` is a `f64` describing how far from perfect the
    /// combination is. `f` is often supplied with the use of the `ROpBuilder` struct.
    pub fn calc(&self, f: impl Fn(&RSet) -> Option<f64>) -> Option<RRes> {
        let mut res: Vec<(u64, RSet)> = self
            .rs
            .iter()
            .map(|r| r.iter().cloned())
            .multi_cartesian_product()
            .filter_map(|v| {
                let rs = RSet(v.into_boxed_slice());
                f(&rs).map(|err| ((err * 1e9).round() as u64, rs))
            })
            .collect();
        res.sort_by_key(|(err, _rs)| *err);
        if !res.is_empty() {
            Some(RRes { res })
        } else {
            None
        }
    }
}
