use itertools::Itertools;
use ndarray::{ArrayView, Ix1, Ix2, Zip};
use std::collections::HashMap;
use std::hash::Hash;

#[derive(Debug, PartialEq)]
pub struct Rule<A, C> {
    pub cases: Vec<Case<A, C>>,
    pub accuracy: Accuracy,
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Case<A, C> {
    pub attribute_value: A,
    pub predicted_class: C,
}

pub fn interpret<'c, A: PartialEq, C>(
    cases: &'c Vec<Case<A, C>>,
    attribute_value: &A,
) -> Option<&'c C> {
    cases
        .iter()
        .find(|case| &case.attribute_value == attribute_value)
        .map(|case| &case.predicted_class)
}

#[derive(Debug, PartialEq, PartialOrd)]
pub struct Accuracy(pub f64);

pub fn evaluate<A: PartialEq, C: PartialEq>(
    cases: &Vec<Case<A, C>>,
    attribute_values: &ArrayView<A, Ix1>,
    classes: &ArrayView<C, Ix1>,
) -> Accuracy {
    let mut right_wrong: Vec<Option<bool>> = Vec::new();

    Zip::from(attribute_values)
        .and(classes)
        .apply(
            |attribute_value, class| match interpret(cases, attribute_value) {
                None => right_wrong.push(None),
                Some(predicted) => right_wrong.push(Some(predicted == class)),
            },
        );

    let num_examples = classes.len();

    if num_examples == 0 {
        Accuracy(0.0)
    } else {
        let num_correct = right_wrong.into_iter().filter(|&o| o == Some(true)).count();
        Accuracy(num_correct as f64 / num_examples as f64)
    }
}

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
/// # Examples
///
/// ```
/// use ndarray::prelude::*;
/// use oner_induction::{Rule, Case, Accuracy, discover};
///
/// let examples = array![
///    ["sunny", "summer"],
///    ["sunny", "summer"],
///    ["cloudy", "winter"],
///    ["sunny", "winter"]
/// ];
///
///
/// let classes = array![
///     "hot",
///     "hot",
///     "cold",
///     "cold"
/// ];
///
/// type ColumnNumber = usize;
/// let rule: Option<(ColumnNumber, Rule<&str, &str>)> =
///   discover(&examples.view(), &classes.view());
///
/// let accuracy = Accuracy(1.0);
///
/// let cases = vec![
///     Case { attribute_value: "summer", predicted_class: "hot" },
///     Case { attribute_value: "winter", predicted_class: "cold" }
/// ];
///
/// // Column 1 is season (winter or summer)
/// assert_eq!(rule, Some( (1, Rule { cases, accuracy }) ));
/// ```
pub fn discover<A, C>(
    attributes: &ArrayView<A, Ix2>,
    classes: &ArrayView<C, Ix1>,
) -> Option<(usize, Rule<A, C>)>
where
    A: Eq + Hash + Clone + std::fmt::Debug,
    C: Eq + Hash + Clone + std::fmt::Debug,
{
    let rules: Vec<Rule<A, C>> = generate_hypotheses(attributes, classes);

    for rule in rules.iter() {
        println!("{:?}", rule);
    }

    let best_rule_and_index = rules.into_iter().enumerate().max_by(|(_i, a), (_j, b)| {
        a.accuracy
            .partial_cmp(&b.accuracy)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println!("Best: {:?}", &best_rule_and_index);

    best_rule_and_index
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::prelude::*;
    #[test]
    fn test1() {
        let attributes = array![
            // location,size,pets
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

        let classes = array![
            "high", "high", "high", "medium", "medium", "medium", "medium", "low", "low", "low",
        ];
        let r = discover(&attributes.view(), &classes.view());
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
        // Count the number of times we see each class:
        let mut class_count: HashMap<&C, i32> = HashMap::new();
        Zip::from(attribute_values)
            .and(classes)
            .apply(|attribute_value, class| {
                if attribute_value == v {
                    *class_count.entry(class).or_insert(0) += 1;
                }
            });

        // The most frequent class is the preidction for the attribute value, v
        let maybe_most_frequent_class = class_count
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(class, _)| class);

        if let Some(class) = maybe_most_frequent_class {
            cases.push(Case {
                attribute_value: v.to_owned(),
                predicted_class: class.to_owned(),
            });
        }
    }

    let accuracy = evaluate(&cases, attribute_values, classes);

    Rule { cases, accuracy }
}
