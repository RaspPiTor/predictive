#![feature(core_intrinsics)]
extern crate rand;
use rand::{thread_rng, Rng};

struct ML {
    input_size: usize,
    output_size: usize,
    nn: Vec<f32>,
    rng: rand::ThreadRng,
}
impl ML {
    pub fn new(input_size: usize, output_size: usize) -> ML {
        let mut new: ML = ML {
            input_size: input_size,
            output_size: output_size,
            nn: Vec::with_capacity(output_size * input_size),
            rng: thread_rng(),
        };
        for _ in 0..(output_size * input_size) {
            new.nn.push(0.0);
        }
        new.randomise();
        new
    }
    fn randomise(&mut self) {
        for i in 0..self.output_size {
            for x in 0..self.input_size {
                self.nn[i * self.output_size + x] = self.rng.gen_range(-4.0, 4.0);
            }
        }
    }
    pub fn predict(&self, input: &Vec<f32>) -> Vec<f32> {
        let mut output: Vec<f32> = Vec::with_capacity(self.output_size);
        for i in 0..self.output_size {
            let mut total: f32 = 0.0;
            for x in 0..self.input_size {
                unsafe {
                    total = std::intrinsics::fadd_fast(
                        total,
                        std::intrinsics::fmul_fast(self.nn[i * self.output_size + x], input[x]),
                    );
                    ;
                }
            }
            output.push(unsafe {
                std::intrinsics::fdiv_fast(fast_math::atan(total), std::f32::consts::FRAC_PI_2)
            });
        }
        return output;
    }
    pub fn evaluate(&self, training_data: &Vec<Vec<Vec<f32>>>) -> f32 {
        let mut total_error: f32 = 0.0;
        for row in 0..training_data.len() {
            let predicted: Vec<f32> = self.predict(&training_data[row][0]);
            for i in 0..self.output_size {
                unsafe {
                    total_error = std::intrinsics::fadd_fast(
                        total_error,
                        std::intrinsics::fsub_fast(predicted[i], training_data[row][1][i]).abs(),
                    );
                }
            }
        }
        unsafe {
            std::intrinsics::fdiv_fast(total_error, (training_data.len() * self.output_size) as f32)
        }
    }
    pub fn optimise_current(&mut self, training_data: &Vec<Vec<Vec<f32>>>, rounds: u32) {
        let mut previous_score: f32 = self.evaluate(&training_data);
        for _ in 0..rounds {
            let mut best_change_location: [usize; 2] = [0; 2];
            let mut best_change: f32 = 0.0;
            let mut new_score: f32 = previous_score + 1.0;
            for i in 0..self.output_size {
                for x in 0..self.input_size {
                    for change in [-0.0001, 0.0001].iter() {
                        let old: f32 = self.nn[i * self.output_size + x];
                        self.nn[i * self.output_size + x] = unsafe {
                            std::intrinsics::fadd_fast(self.nn[i * self.output_size + x], *change)
                        };
                        let current_score: f32 = self.evaluate(&training_data);
                        if { current_score < new_score } {
                            new_score = current_score;
                            best_change = *change;
                            best_change_location = [i, x];
                        }
                        self.nn[i * self.output_size + x] = old;
                    }
                }
            }
            if { new_score < previous_score } {
                self.nn[best_change_location[0] * self.output_size + best_change_location[1]] = unsafe {
                    std::intrinsics::fadd_fast(
                        self.nn
                            [best_change_location[0] * self.output_size + best_change_location[1]],
                        best_change,
                    )
                };
                previous_score = new_score;
            } else {
                return;
            }
        }
        println!("Ran out of rounds with max of: {:?}", rounds);
    }
    pub fn train(&mut self, training_data: &Vec<Vec<Vec<f32>>>) {
        let mut best: Vec<f32> = self.nn.clone();
        let mut best_score: f32 = self.evaluate(&training_data);
        for round in 0..(1000 * 1000 * 1000) {
            self.randomise();
            self.optimise_current(&training_data, 1000 * 1000 * 1000);
            let score: f32 = self.evaluate(&training_data);
            if { score < best_score } {
                best_score = score;
                best = self.nn.clone();
                println!(
                    "Round {:?}, new score: {:?}, nn: {:?}",
                    round, score, self.nn
                );
            } else {
                println!("Round {:?}", round);
            }
        }
        self.nn = best;
    }
}
fn main() {
    let mut the_machine: ML = ML::new(4, 2);
    println!("{:?}", the_machine.nn);
    println!("{:?}", the_machine.predict(&vec![1.0, 2.0, 3.0, 4.0]));
    let old_score: f32 =
        the_machine.evaluate(&vec![vec![vec![1.0, 2.0, 3.0, 4.0], vec![0.0, 1.0]]]);
    println!("{:?}", old_score);
    the_machine.train(&vec![vec![vec![1.0, 2.0, 3.0, 4.0], vec![0.0, 1.0]]]);
}
