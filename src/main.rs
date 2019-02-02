use predictive;
fn main() {
    let mut the_machine: ML = ML::new(4, 5, 5, 5);
    the_machine.train(
        &vec![
            vec![vec![4.0, 3.0, 2.0, 1.0], vec![1.0, 0.75, 0.5, 0.25, 0.0]],
        ],
        10000,
    );
    println!(
        "[1.0, 2.0, 3.0, 4.0]:{:?}, [4.0, 3.0, 2.0, 1.0]:{:?}",
        the_machine.predict_public(&vec![1.0, 2.0, 3.0, 4.0]),
        the_machine.predict_public(&vec![4.0, 3.0, 2.0, 1.0])
    )
}
