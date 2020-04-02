// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use super::evaluation::evaluate;
use super::{Case, Rule};
use itertools::Itertools;
use ndarray::{ArrayView, Ix1, Ix2, Zip};
use rustc_hash::FxHasher;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hash};

/// Find the one rule that fits a set of example data points.
///
/// # Arguments
///
/// * `attributes` - rows containing attribute values as columns.
/// * `classes` - the true classification for each row.
///
/// # Result
///
/// The result is a tuple containing the column index the best rule applies to,
/// and the `Rule` itself.
///
/// The `Rule` consists of:
/// - the attribute index (from zero) that the rule works for; and
/// - the `Case`s for that attribute.
///
/// A `Case` is a value for the attribute and the corresponding predicted class.
///
pub fn discover<A, C>(
    attributes: &ArrayView<A, Ix2>,
    classes: &ArrayView<C, Ix1>,
) -> Option<(usize, Rule<A, C>)>
where
    A: Eq + Hash + Clone + std::fmt::Debug,
    C: Eq + Hash + Clone + std::fmt::Debug,
{
    let rules: Vec<Rule<A, C>> = generate_hypotheses(attributes, classes);

    // Find the best rule (highest accuracy), and the column number it applies to:
    rules.into_iter().enumerate().max_by(|(_i, a), (_j, b)| {
        a.accuracy.partial_cmp(&b.accuracy).unwrap_or(std::cmp::Ordering::Equal)
    })
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::Accuracy;
    use ndarray::prelude::*;
    #[test]
    fn test1() {
        let attributes = array![
            // Data from: Christoph Molnar's "Interpretable Machine Learning",
            // licensed under https://creativecommons.org/licenses/by-nc-sa/4.0/
            // rental property attributes: location,size,pets
            ["good", "small", "yes"],
            ["good", "big", "no"],
            ["good", "big", "no"],
            ["bad", "medium", "no"],
            ["good", "medium", "only cats"],
            ["good", "small", "only cats"],
            ["bad", "medium", "yes"],
            ["bad", "small", "yes"],
            ["bad", "medium", "yes"],
            ["bad", "small", "no"],
        ];

        // rental property value:
        let classes = array![
            "high", "high", "high", "medium", "medium", "medium", "medium", "low", "low", "low",
        ];

        let rule = discover(&attributes.view(), &classes.view());

        let expected_rule = Rule {
            cases: vec![
                Case { attribute_value: "small", predicted_class: "low" },
                Case { attribute_value: "big", predicted_class: "high" },
                Case { attribute_value: "medium", predicted_class: "medium" },
            ],
            accuracy: Accuracy(0.7),
        };

        assert_eq!(rule, Some((1, expected_rule)));
    }
}

fn generate_hypotheses<A: Eq + Hash + Clone, C: Eq + Hash + Clone>(
    attributes: &ArrayView<A, Ix2>,
    classes: &ArrayView<C, Ix1>,
) -> Vec<Rule<A, C>> {
    let mut hs = Vec::new();

    // Generate a rule for each attribute:
    for col in attributes.gencolumns() {
        let hypothesis = generate_rule_for_attribute(&col, classes);
        hs.push(hypothesis);
    }

    hs
}

/// Generate a rule based on a single attribute.
///
/// The process works by finding the most frequent class for each distinct
/// attribute value. The most frequent class is the prediction for that attribute value.
///
/// The result is a set of "cases" (one "IF ... THEN" condition for each distinct attribute value).
///
/// # Arguments
///
/// * `attribute_values` - the value of each attribute for all examples being used.
///
/// * `clases` - the true value for each class.
///
/// These arguments are the training data for the rule.
/// The arguments must be of the same length. For each attribute value, there's a corresponding class.
///
fn generate_rule_for_attribute<A, C>(
    attribute_values: &ArrayView<A, Ix1>,
    classes: &ArrayView<C, Ix1>,
) -> Rule<A, C>
where
    A: Eq + Hash + Clone,
    C: Eq + Hash + Clone,
{
    let mut cases: Vec<Case<A, C>> = Vec::new();

    let unique_values = attribute_values.iter().unique();

    for v in unique_values {
        // Count the number of times we see each class, using deterministic hasher for reproducabiltiy with tied results
        let mut class_count = HashMap::with_hasher(BuildHasherDefault::<FxHasher>::default());
        Zip::from(attribute_values).and(classes).apply(|attribute_value, class| {
            if attribute_value == v {
                *class_count.entry(class).or_insert(0) += 1;
            }
        });

        // The most frequent class is the preidction for the attribute value, v
        let maybe_most_frequent_class =
            class_count.into_iter().max_by_key(|&(_, count)| count).map(|(class, _)| class);

        if let Some(class) = maybe_most_frequent_class {
            cases.push(Case { attribute_value: v.to_owned(), predicted_class: class.to_owned() });
        }
    }

    let accuracy = evaluate(&cases, attribute_values, classes);

    Rule { cases, accuracy }
}
