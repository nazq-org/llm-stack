//! Token usage and cost tracking.
//!
//! Every response carries a [`Usage`] record counting input and output
//! tokens, with optional fields for reasoning and cache tokens when the
//! provider reports them.
//!
//! [`Cost`] tracks monetary cost in **microdollars** (1 USD = 1,000,000
//! microdollars). Integer arithmetic avoids floating-point rounding
//! issues when aggregating costs across many requests. Use
//! [`total_usd`](Cost::total_usd) for display purposes.
//!
//! # Invariant
//!
//! `Cost` enforces `total == input + output` at construction time.
//! The fields are private — use [`Cost::new`] to build one, and the
//! accessor methods to read values. Deserialization recomputes the
//! total from `input` and `output`, ignoring any `total` in the JSON.

use std::fmt;
use std::ops::{Add, AddAssign};

use serde::{Deserialize, Serialize};

/// Token counts for a single request/response pair.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Usage {
    /// Tokens consumed by the prompt (messages + system + tool defs).
    pub input_tokens: u64,
    /// Tokens produced by the model's response.
    pub output_tokens: u64,
    /// Tokens used for chain-of-thought reasoning, if applicable.
    pub reasoning_tokens: Option<u64>,
    /// Tokens served from the provider's prompt cache (reducing cost).
    pub cache_read_tokens: Option<u64>,
    /// Tokens written into the provider's prompt cache for future reuse.
    pub cache_write_tokens: Option<u64>,
}

/// Helper: adds two `Option<u64>` fields, treating `None` as zero.
fn add_optional(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.saturating_add(y)),
        (Some(x), None) | (None, Some(x)) => Some(x),
        (None, None) => None,
    }
}

impl Add for Usage {
    type Output = Self;

    /// Adds two `Usage` records field-by-field.
    ///
    /// Mandatory fields use saturating addition. Optional fields are
    /// summed when both are `Some`, preserved when one is `Some`, and
    /// remain `None` when both are `None`.
    fn add(self, rhs: Self) -> Self {
        Self {
            input_tokens: self.input_tokens.saturating_add(rhs.input_tokens),
            output_tokens: self.output_tokens.saturating_add(rhs.output_tokens),
            reasoning_tokens: add_optional(self.reasoning_tokens, rhs.reasoning_tokens),
            cache_read_tokens: add_optional(self.cache_read_tokens, rhs.cache_read_tokens),
            cache_write_tokens: add_optional(self.cache_write_tokens, rhs.cache_write_tokens),
        }
    }
}

impl AddAssign for Usage {
    fn add_assign(&mut self, rhs: Self) {
        *self += &rhs;
    }
}

impl AddAssign<&Usage> for Usage {
    /// Adds another `Usage` to this one in-place without cloning.
    ///
    /// This is more efficient than `AddAssign<Usage>` when you have a reference.
    fn add_assign(&mut self, rhs: &Self) {
        self.input_tokens = self.input_tokens.saturating_add(rhs.input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(rhs.output_tokens);
        self.reasoning_tokens = add_optional(self.reasoning_tokens, rhs.reasoning_tokens);
        self.cache_read_tokens = add_optional(self.cache_read_tokens, rhs.cache_read_tokens);
        self.cache_write_tokens = add_optional(self.cache_write_tokens, rhs.cache_write_tokens);
    }
}

/// Monetary cost in microdollars (1 USD = 1,000,000 microdollars).
///
/// Uses integer arithmetic to avoid floating-point accumulation errors.
/// The invariant `total == input + output` is enforced by the
/// constructor and maintained through deserialization.
///
/// # Examples
///
/// ```rust
/// use llm_stack_core::Cost;
///
/// let cost = Cost::new(300_000, 150_000).expect("no overflow");
/// assert_eq!(cost.total_microdollars(), 450_000);
/// assert!((cost.total_usd() - 0.45).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Cost {
    input: u64,
    output: u64,
    total: u64,
}

impl Default for Cost {
    /// Returns a zero cost.
    fn default() -> Self {
        Self {
            input: 0,
            output: 0,
            total: 0,
        }
    }
}

/// Intermediate type for safe deserialization — recomputes total.
#[derive(Deserialize)]
struct CostRaw {
    input: u64,
    output: u64,
}

impl<'de> Deserialize<'de> for Cost {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = CostRaw::deserialize(deserializer)?;
        let total = raw
            .input
            .checked_add(raw.output)
            .ok_or_else(|| serde::de::Error::custom("cost overflow: input + output exceeds u64"))?;
        Ok(Self {
            input: raw.input,
            output: raw.output,
            total,
        })
    }
}

impl Cost {
    /// Creates a new `Cost`, returning `None` if `input + output`
    /// would overflow `u64`.
    pub fn new(input: u64, output: u64) -> Option<Self> {
        let total = input.checked_add(output)?;
        Some(Self {
            input,
            output,
            total,
        })
    }

    /// Cost of the input (prompt) in microdollars.
    pub fn input_microdollars(&self) -> u64 {
        self.input
    }

    /// Cost of the output (completion) in microdollars.
    pub fn output_microdollars(&self) -> u64 {
        self.output
    }

    /// Total cost (`input + output`) in microdollars.
    pub fn total_microdollars(&self) -> u64 {
        self.total
    }

    /// Returns the sum of two costs, or `None` on overflow.
    pub fn checked_add(&self, rhs: &Self) -> Option<Self> {
        let input = self.input.checked_add(rhs.input)?;
        let output = self.output.checked_add(rhs.output)?;
        Self::new(input, output)
    }

    /// Total cost in US dollars, for display purposes.
    ///
    /// Uses floating-point division — prefer
    /// [`total_microdollars`](Self::total_microdollars) for arithmetic.
    #[allow(clippy::cast_precision_loss)] // microdollar u64 fits f64 mantissa in practice
    pub fn total_usd(&self) -> f64 {
        self.total as f64 / 1_000_000.0
    }
}

impl fmt::Display for Cost {
    /// Formats the cost as a USD string, e.g. `$1.50`.
    #[allow(clippy::cast_precision_loss)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "${:.2}", self.total as f64 / 1_000_000.0)
    }
}

impl Add for Cost {
    type Output = Self;

    /// Adds two costs using saturating arithmetic.
    ///
    /// Use [`checked_add`](Self::checked_add) when overflow must be detected.
    fn add(self, rhs: Self) -> Self {
        let input = self.input.saturating_add(rhs.input);
        let output = self.output.saturating_add(rhs.output);
        Self {
            input,
            output,
            total: input.saturating_add(output),
        }
    }
}

impl AddAssign for Cost {
    fn add_assign(&mut self, rhs: Self) {
        self.input = self.input.saturating_add(rhs.input);
        self.output = self.output.saturating_add(rhs.output);
        self.total = self.input.saturating_add(self.output);
    }
}

// ── UsageTracker ────────────────────────────────────────────────────

/// Tracks cumulative token usage across multiple LLM calls.
///
/// `UsageTracker` accumulates [`Usage`] records from each request and
/// provides context-awareness features for detecting when the conversation
/// is approaching the model's context limit.
///
/// # Example
///
/// ```rust
/// use llm_stack_core::usage::{Usage, UsageTracker};
///
/// let mut tracker = UsageTracker::with_context_limit(128_000);
///
/// // Record usage from each LLM call
/// tracker.record(Usage {
///     input_tokens: 1000,
///     output_tokens: 500,
///     ..Default::default()
/// });
///
/// assert_eq!(tracker.total().input_tokens, 1000);
/// assert!(!tracker.is_near_limit(0.8)); // Not near 80% yet
/// ```
///
/// # Use Cases
///
/// - **Billing/cost tracking**: Aggregate costs across a session
/// - **Budget alerts**: Warn when approaching token limits
/// - **Compaction triggers**: Signal when context window is nearly full
/// - **Token debugging**: Analyze per-call consumption patterns
#[derive(Debug, Clone)]
pub struct UsageTracker {
    /// Accumulated usage across all calls.
    total: Usage,
    /// Usage from each individual call, in order.
    by_call: Vec<Usage>,
    /// Optional context window limit for utilization calculations.
    context_limit: Option<u64>,
}

impl Default for UsageTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl UsageTracker {
    /// Creates a new tracker with no context limit.
    pub fn new() -> Self {
        Self {
            total: Usage::default(),
            by_call: Vec::new(),
            context_limit: None,
        }
    }

    /// Creates a tracker with a known context window limit.
    ///
    /// The limit is used for [`context_utilization`](Self::context_utilization)
    /// and [`is_near_limit`](Self::is_near_limit) calculations.
    pub fn with_context_limit(limit: u64) -> Self {
        Self {
            total: Usage::default(),
            by_call: Vec::new(),
            context_limit: Some(limit),
        }
    }

    /// Records a usage sample from an LLM call.
    ///
    /// The usage is added to the running total and stored for per-call
    /// analysis.
    pub fn record(&mut self, usage: Usage) {
        self.total += &usage;
        self.by_call.push(usage);
    }

    /// Returns the accumulated usage across all recorded calls.
    pub fn total(&self) -> &Usage {
        &self.total
    }

    /// Returns the usage from each individual call, in order.
    pub fn calls(&self) -> &[Usage] {
        &self.by_call
    }

    /// Returns the number of calls recorded.
    pub fn call_count(&self) -> usize {
        self.by_call.len()
    }

    /// Returns the context limit, if set.
    pub fn context_limit(&self) -> Option<u64> {
        self.context_limit
    }

    /// Sets or updates the context limit.
    ///
    /// Useful when the model is determined after tracker creation.
    pub fn set_context_limit(&mut self, limit: u64) {
        self.context_limit = Some(limit);
    }

    /// Returns the context utilization as a ratio (0.0 to 1.0+).
    ///
    /// Utilization is calculated as `total_input_tokens / context_limit`.
    /// Returns `None` if no context limit is set.
    ///
    /// # Note
    ///
    /// The value can exceed 1.0 if the total exceeds the limit (which
    /// shouldn't happen in practice but is not enforced).
    #[allow(clippy::cast_precision_loss)] // u64 token counts fit f64 mantissa
    pub fn context_utilization(&self) -> Option<f64> {
        self.context_limit.map(|limit| {
            if limit == 0 {
                return 0.0;
            }
            self.total.input_tokens as f64 / limit as f64
        })
    }

    /// Checks if the context utilization is at or above the given threshold.
    ///
    /// Returns `false` if no context limit is set.
    ///
    /// # Example
    ///
    /// ```rust
    /// use llm_stack_core::usage::{Usage, UsageTracker};
    ///
    /// let mut tracker = UsageTracker::with_context_limit(100_000);
    /// tracker.record(Usage {
    ///     input_tokens: 85_000,
    ///     output_tokens: 1000,
    ///     ..Default::default()
    /// });
    ///
    /// assert!(tracker.is_near_limit(0.8));   // 85% >= 80%
    /// assert!(!tracker.is_near_limit(0.9));  // 85% < 90%
    /// ```
    pub fn is_near_limit(&self, threshold: f64) -> bool {
        self.context_utilization()
            .is_some_and(|util| util >= threshold)
    }

    /// Computes the cost of all recorded usage given a pricing table.
    ///
    /// Uses the pricing rates (per-million tokens) to calculate cost.
    /// Returns `None` if the cost would overflow.
    pub fn cost(&self, pricing: &ModelPricing) -> Option<Cost> {
        pricing.compute_cost(&self.total)
    }

    /// Resets the tracker, clearing all recorded usage.
    pub fn reset(&mut self) {
        self.total = Usage::default();
        self.by_call.clear();
    }
}

/// Pricing information for a specific model.
///
/// All prices are in **microdollars per million tokens**. For example,
/// a price of $3.00 per million input tokens would be `3_000_000`.
///
/// # Example
///
/// ```rust
/// use llm_stack_core::usage::{ModelPricing, Usage};
///
/// // Claude 3.5 Sonnet pricing (as of early 2024)
/// let pricing = ModelPricing {
///     input_per_million: 3_000_000,   // $3.00 / MTok
///     output_per_million: 15_000_000, // $15.00 / MTok
///     cache_read_per_million: Some(300_000), // $0.30 / MTok
/// };
///
/// let usage = Usage {
///     input_tokens: 1_000_000,
///     output_tokens: 100_000,
///     ..Default::default()
/// };
///
/// let cost = pricing.compute_cost(&usage).unwrap();
/// assert_eq!(cost.total_microdollars(), 4_500_000); // $3 input + $1.50 output
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelPricing {
    /// Cost per million input tokens in microdollars.
    pub input_per_million: u64,
    /// Cost per million output tokens in microdollars.
    pub output_per_million: u64,
    /// Cost per million cache-read tokens in microdollars (if applicable).
    pub cache_read_per_million: Option<u64>,
}

impl ModelPricing {
    /// Computes the cost for the given usage.
    ///
    /// Returns `None` if the calculation would overflow.
    pub fn compute_cost(&self, usage: &Usage) -> Option<Cost> {
        // Cost = (tokens * price_per_million) / 1_000_000
        // To avoid precision loss, we use u128 for intermediate calculations
        let input_cost = compute_token_cost(usage.input_tokens, self.input_per_million)?;
        let output_cost = compute_token_cost(usage.output_tokens, self.output_per_million)?;

        // If cache read tokens are present and pricing is set, include them
        let cache_cost = match (usage.cache_read_tokens, self.cache_read_per_million) {
            (Some(tokens), Some(rate)) => compute_token_cost(tokens, rate)?,
            _ => 0,
        };

        // Combine costs (cache reads reduce effective input cost conceptually,
        // but for billing they're additive at the cache rate)
        let total_input = input_cost.checked_add(cache_cost)?;
        Cost::new(total_input, output_cost)
    }
}

/// Compute cost for a token count at a given rate.
///
/// Returns microdollars, or `None` on overflow.
fn compute_token_cost(tokens: u64, per_million: u64) -> Option<u64> {
    // (tokens * per_million) / 1_000_000
    // Use u128 to avoid overflow in multiplication
    let product = u128::from(tokens) * u128::from(per_million);
    let cost = product / 1_000_000;
    u64::try_from(cost).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_usage_clone_eq() {
        let u = Usage {
            input_tokens: 100,
            output_tokens: 50,
            reasoning_tokens: Some(10),
            cache_read_tokens: None,
            cache_write_tokens: None,
        };
        assert_eq!(u, u.clone());
    }

    #[test]
    fn test_usage_debug_format() {
        let u = Usage::default();
        let debug = format!("{u:?}");
        assert!(debug.contains("input_tokens"));
        assert!(debug.contains("output_tokens"));
    }

    #[test]
    fn test_usage_optional_fields_none() {
        let u = Usage::default();
        assert_eq!(u.reasoning_tokens, None);
        assert_eq!(u.cache_read_tokens, None);
        assert_eq!(u.cache_write_tokens, None);
    }

    #[test]
    fn test_usage_optional_fields_some() {
        let u = Usage {
            input_tokens: 0,
            output_tokens: 0,
            reasoning_tokens: Some(500),
            cache_read_tokens: Some(200),
            cache_write_tokens: Some(100),
        };
        assert_eq!(u.reasoning_tokens, Some(500));
        assert_eq!(u.cache_read_tokens, Some(200));
        assert_eq!(u.cache_write_tokens, Some(100));
    }

    #[test]
    fn test_usage_serde_roundtrip() {
        let u = Usage {
            input_tokens: 100,
            output_tokens: 50,
            reasoning_tokens: Some(10),
            cache_read_tokens: None,
            cache_write_tokens: None,
        };
        let json = serde_json::to_string(&u).unwrap();
        let back: Usage = serde_json::from_str(&json).unwrap();
        assert_eq!(u, back);
    }

    #[test]
    fn test_cost_new_enforces_invariant() {
        let c = Cost::new(1_000_000, 500_000).unwrap();
        assert_eq!(c.input_microdollars(), 1_000_000);
        assert_eq!(c.output_microdollars(), 500_000);
        assert_eq!(c.total_microdollars(), 1_500_000);
    }

    #[test]
    fn test_cost_new_overflow_returns_none() {
        assert!(Cost::new(u64::MAX, 1).is_none());
    }

    #[test]
    fn test_cost_total_usd_exact() {
        let c = Cost::new(1_000_000, 500_000).unwrap();
        assert!((c.total_usd() - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cost_total_usd_zero() {
        let c = Cost::new(0, 0).unwrap();
        assert!((c.total_usd()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cost_total_usd_sub_cent() {
        let c = Cost::new(300, 200).unwrap();
        assert!((c.total_usd() - 0.0005).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cost_clone_eq() {
        let c = Cost::new(42, 58).unwrap();
        assert_eq!(c, c.clone());
    }

    #[test]
    fn test_cost_serde_roundtrip() {
        let c = Cost::new(1_000_000, 500_000).unwrap();
        let json = serde_json::to_string(&c).unwrap();
        let back: Cost = serde_json::from_str(&json).unwrap();
        assert_eq!(c, back);
    }

    #[test]
    fn test_cost_deserialization_recomputes_total() {
        // Even if JSON has a wrong total, deserialization recomputes it
        let json = r#"{"input":100,"output":200,"total":999}"#;
        let c: Cost = serde_json::from_str(json).unwrap();
        assert_eq!(c.total_microdollars(), 300);
    }

    #[test]
    fn test_cost_deserialization_without_total() {
        let json = r#"{"input":100,"output":200}"#;
        let c: Cost = serde_json::from_str(json).unwrap();
        assert_eq!(c.total_microdollars(), 300);
    }

    #[test]
    fn test_cost_deserialization_overflow_fails() {
        let json = format!(r#"{{"input":{},"output":1}}"#, u64::MAX);
        let result: Result<Cost, _> = serde_json::from_str(&json);
        assert!(result.is_err());
    }

    #[test]
    fn test_cost_default_is_zero() {
        let c = Cost::default();
        assert_eq!(c.input_microdollars(), 0);
        assert_eq!(c.output_microdollars(), 0);
        assert_eq!(c.total_microdollars(), 0);
    }

    // --- Cost Display ---

    #[test]
    fn test_cost_display() {
        let c = Cost::new(1_000_000, 500_000).unwrap();
        assert_eq!(c.to_string(), "$1.50");
    }

    #[test]
    fn test_cost_display_zero() {
        assert_eq!(Cost::default().to_string(), "$0.00");
    }

    #[test]
    fn test_cost_display_sub_cent() {
        let c = Cost::new(500, 0).unwrap();
        assert_eq!(c.to_string(), "$0.00");
    }

    // --- Usage Add/AddAssign ---

    #[test]
    fn test_usage_add_basic() {
        let a = Usage {
            input_tokens: 100,
            output_tokens: 50,
            reasoning_tokens: Some(10),
            cache_read_tokens: None,
            cache_write_tokens: Some(20),
        };
        let b = Usage {
            input_tokens: 200,
            output_tokens: 30,
            reasoning_tokens: Some(5),
            cache_read_tokens: Some(50),
            cache_write_tokens: None,
        };
        let sum = a + b;
        assert_eq!(sum.input_tokens, 300);
        assert_eq!(sum.output_tokens, 80);
        assert_eq!(sum.reasoning_tokens, Some(15));
        assert_eq!(sum.cache_read_tokens, Some(50));
        assert_eq!(sum.cache_write_tokens, Some(20));
    }

    #[test]
    fn test_usage_add_both_none() {
        let a = Usage::default();
        let b = Usage::default();
        let sum = a + b;
        assert_eq!(sum.reasoning_tokens, None);
        assert_eq!(sum.cache_read_tokens, None);
        assert_eq!(sum.cache_write_tokens, None);
    }

    #[test]
    fn test_usage_add_assign() {
        let mut a = Usage {
            input_tokens: 100,
            output_tokens: 50,
            ..Default::default()
        };
        a += Usage {
            input_tokens: 200,
            output_tokens: 30,
            ..Default::default()
        };
        assert_eq!(a.input_tokens, 300);
        assert_eq!(a.output_tokens, 80);
    }

    #[test]
    fn test_usage_add_saturates() {
        let a = Usage {
            input_tokens: u64::MAX,
            output_tokens: 0,
            ..Default::default()
        };
        let b = Usage {
            input_tokens: 1,
            output_tokens: 0,
            ..Default::default()
        };
        let sum = a + b;
        assert_eq!(sum.input_tokens, u64::MAX);
    }

    // --- Cost Add/AddAssign/checked_add ---

    #[test]
    fn test_cost_add_basic() {
        let a = Cost::new(100, 200).unwrap();
        let b = Cost::new(300, 400).unwrap();
        let sum = a + b;
        assert_eq!(sum.input_microdollars(), 400);
        assert_eq!(sum.output_microdollars(), 600);
        assert_eq!(sum.total_microdollars(), 1000);
    }

    #[test]
    fn test_cost_add_assign() {
        let mut c = Cost::new(100, 200).unwrap();
        c += Cost::new(50, 50).unwrap();
        assert_eq!(c.input_microdollars(), 150);
        assert_eq!(c.output_microdollars(), 250);
        assert_eq!(c.total_microdollars(), 400);
    }

    #[test]
    fn test_cost_checked_add() {
        let a = Cost::new(100, 200).unwrap();
        let b = Cost::new(300, 400).unwrap();
        let sum = a.checked_add(&b).unwrap();
        assert_eq!(sum.total_microdollars(), 1000);
    }

    #[test]
    fn test_cost_checked_add_overflow() {
        let a = Cost::new(u64::MAX - 1, 0).unwrap();
        let b = Cost::new(2, 0).unwrap();
        assert!(a.checked_add(&b).is_none());
    }

    #[test]
    fn test_cost_add_saturates() {
        let a = Cost::new(u64::MAX - 1, 0).unwrap();
        let b = Cost::new(2, 0).unwrap();
        let sum = a + b;
        assert_eq!(sum.input_microdollars(), u64::MAX);
    }

    // --- UsageTracker ---

    #[test]
    fn test_usage_tracker_new() {
        let tracker = UsageTracker::new();
        assert_eq!(tracker.total().input_tokens, 0);
        assert_eq!(tracker.total().output_tokens, 0);
        assert!(tracker.calls().is_empty());
        assert_eq!(tracker.context_limit(), None);
    }

    #[test]
    fn test_usage_tracker_default() {
        let tracker = UsageTracker::default();
        assert_eq!(tracker.call_count(), 0);
        assert_eq!(tracker.context_limit(), None);
    }

    #[test]
    fn test_usage_tracker_with_context_limit() {
        let tracker = UsageTracker::with_context_limit(128_000);
        assert_eq!(tracker.context_limit(), Some(128_000));
    }

    #[test]
    fn test_usage_tracker_record() {
        let mut tracker = UsageTracker::new();
        tracker.record(Usage {
            input_tokens: 100,
            output_tokens: 50,
            ..Default::default()
        });
        tracker.record(Usage {
            input_tokens: 200,
            output_tokens: 100,
            ..Default::default()
        });

        assert_eq!(tracker.total().input_tokens, 300);
        assert_eq!(tracker.total().output_tokens, 150);
        assert_eq!(tracker.call_count(), 2);
        assert_eq!(tracker.calls()[0].input_tokens, 100);
        assert_eq!(tracker.calls()[1].input_tokens, 200);
    }

    #[test]
    fn test_usage_tracker_context_utilization() {
        let mut tracker = UsageTracker::with_context_limit(100_000);
        tracker.record(Usage {
            input_tokens: 50_000,
            output_tokens: 1000,
            ..Default::default()
        });

        let util = tracker.context_utilization().unwrap();
        assert!((util - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_usage_tracker_context_utilization_no_limit() {
        let tracker = UsageTracker::new();
        assert!(tracker.context_utilization().is_none());
    }

    #[test]
    fn test_usage_tracker_context_utilization_zero_limit() {
        let tracker = UsageTracker::with_context_limit(0);
        assert!((tracker.context_utilization().unwrap()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_usage_tracker_is_near_limit() {
        let mut tracker = UsageTracker::with_context_limit(100_000);
        tracker.record(Usage {
            input_tokens: 85_000,
            output_tokens: 1000,
            ..Default::default()
        });

        assert!(tracker.is_near_limit(0.8)); // 85% >= 80%
        assert!(tracker.is_near_limit(0.85)); // 85% >= 85%
        assert!(!tracker.is_near_limit(0.9)); // 85% < 90%
    }

    #[test]
    fn test_usage_tracker_is_near_limit_no_limit() {
        let tracker = UsageTracker::new();
        assert!(!tracker.is_near_limit(0.8));
    }

    #[test]
    fn test_usage_tracker_set_context_limit() {
        let mut tracker = UsageTracker::new();
        assert_eq!(tracker.context_limit(), None);

        tracker.set_context_limit(200_000);
        assert_eq!(tracker.context_limit(), Some(200_000));
    }

    #[test]
    fn test_usage_tracker_reset() {
        let mut tracker = UsageTracker::with_context_limit(100_000);
        tracker.record(Usage {
            input_tokens: 1000,
            output_tokens: 500,
            ..Default::default()
        });
        assert_eq!(tracker.call_count(), 1);
        assert_eq!(tracker.total().input_tokens, 1000);

        tracker.reset();
        assert_eq!(tracker.call_count(), 0);
        assert_eq!(tracker.total().input_tokens, 0);
        // Context limit should be preserved
        assert_eq!(tracker.context_limit(), Some(100_000));
    }

    #[test]
    fn test_usage_tracker_clone() {
        let mut tracker = UsageTracker::with_context_limit(50_000);
        tracker.record(Usage {
            input_tokens: 100,
            output_tokens: 50,
            ..Default::default()
        });

        let cloned = tracker.clone();
        assert_eq!(cloned.total().input_tokens, 100);
        assert_eq!(cloned.call_count(), 1);
        assert_eq!(cloned.context_limit(), Some(50_000));
    }

    // --- ModelPricing ---

    #[test]
    fn test_model_pricing_compute_cost() {
        let pricing = ModelPricing {
            input_per_million: 3_000_000,   // $3 per MTok
            output_per_million: 15_000_000, // $15 per MTok
            cache_read_per_million: None,
        };

        let usage = Usage {
            input_tokens: 1_000_000, // 1 MTok input
            output_tokens: 100_000,  // 0.1 MTok output
            ..Default::default()
        };

        let cost = pricing.compute_cost(&usage).unwrap();
        assert_eq!(cost.input_microdollars(), 3_000_000); // $3
        assert_eq!(cost.output_microdollars(), 1_500_000); // $1.50
        assert_eq!(cost.total_microdollars(), 4_500_000); // $4.50
    }

    #[test]
    fn test_model_pricing_with_cache_tokens() {
        let pricing = ModelPricing {
            input_per_million: 3_000_000,
            output_per_million: 15_000_000,
            cache_read_per_million: Some(300_000), // $0.30 per MTok
        };

        let usage = Usage {
            input_tokens: 500_000,
            output_tokens: 100_000,
            cache_read_tokens: Some(500_000), // 0.5 MTok from cache
            ..Default::default()
        };

        let cost = pricing.compute_cost(&usage).unwrap();
        // Input: 500k * $3/MTok = $1.50 = 1_500_000
        // Cache: 500k * $0.30/MTok = $0.15 = 150_000
        // Total input side: $1.65 = 1_650_000
        // Output: 100k * $15/MTok = $1.50 = 1_500_000
        assert_eq!(cost.input_microdollars(), 1_650_000);
        assert_eq!(cost.output_microdollars(), 1_500_000);
    }

    #[test]
    fn test_model_pricing_zero_tokens() {
        let pricing = ModelPricing {
            input_per_million: 3_000_000,
            output_per_million: 15_000_000,
            cache_read_per_million: None,
        };

        let usage = Usage::default();
        let cost = pricing.compute_cost(&usage).unwrap();
        assert_eq!(cost.total_microdollars(), 0);
    }

    #[test]
    fn test_model_pricing_cache_without_pricing() {
        // Cache tokens present but no cache pricing — should ignore
        let pricing = ModelPricing {
            input_per_million: 3_000_000,
            output_per_million: 15_000_000,
            cache_read_per_million: None,
        };

        let usage = Usage {
            input_tokens: 1_000_000,
            output_tokens: 100_000,
            cache_read_tokens: Some(500_000),
            ..Default::default()
        };

        let cost = pricing.compute_cost(&usage).unwrap();
        // Cache tokens ignored since no pricing
        assert_eq!(cost.input_microdollars(), 3_000_000);
    }

    #[test]
    fn test_usage_tracker_cost() {
        let mut tracker = UsageTracker::new();
        tracker.record(Usage {
            input_tokens: 1_000_000,
            output_tokens: 100_000,
            ..Default::default()
        });

        let pricing = ModelPricing {
            input_per_million: 3_000_000,
            output_per_million: 15_000_000,
            cache_read_per_million: None,
        };

        let cost = tracker.cost(&pricing).unwrap();
        assert_eq!(cost.total_microdollars(), 4_500_000);
    }

    #[test]
    fn test_model_pricing_clone_eq() {
        let p1 = ModelPricing {
            input_per_million: 100,
            output_per_million: 200,
            cache_read_per_million: Some(50),
        };
        let p2 = p1.clone();
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_compute_token_cost_large_values() {
        // Test with large but reasonable values
        let cost = compute_token_cost(10_000_000_000, 3_000_000);
        // 10B tokens * $3/MTok = $30,000 = 30_000_000_000 microdollars
        assert_eq!(cost, Some(30_000_000_000));
    }
}
