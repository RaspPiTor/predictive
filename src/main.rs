#![feature(core_intrinsics)]
extern crate rand;
use rand::{thread_rng, Rng};

struct ML {
    input_size: usize,
    output_size: usize,
    nn: Vec<Vec<f32>>,
    rng: rand::ThreadRng,
}
impl ML {
    pub fn new(input_size: usize, output_size: usize) -> ML {
        let mut new: ML = ML {
            input_size: input_size,
            output_size: output_size,
            nn: Vec::with_capacity(0),
            rng: thread_rng(),
        };
        new.randomise();
        new
    }
    fn randomise(&mut self) {
        let mut new_nn: Vec<Vec<f32>> = Vec::with_capacity(self.output_size);
        let mut i: usize = 0;
        while { i < self.output_size } {
            let mut x: usize = 0;
            let mut new_row: Vec<f32> = Vec::with_capacity(self.input_size);
            while { x < self.input_size } {
                new_row.push(self.rng.gen_range(-2.0, 2.0));
                x += 1;
            }
            new_nn.push(new_row);
            i += 1;
        }
        self.nn = new_nn;
    }
    pub fn predict(&self, input: &Vec<f32>) -> Vec<f32> {
        let mut output: Vec<f32> = Vec::with_capacity(self.output_size);
        let mut i: usize = 0;
        while { i < self.output_size } {
            let mut x: usize = 0;
            let mut total: f32 = 0.0;
            while { x < self.input_size } {
                unsafe {
                    total = std::intrinsics::fadd_fast(
                        total,
                        std::intrinsics::fmul_fast(self.nn[i][x], input[x]),
                    );
                    ;
                }
                x += 1;
            }
            i += 1;
            output.push(unsafe {
                std::intrinsics::fdiv_fast(fast_math::atan(total), std::f32::consts::PI)
            });
        }
        return output;
    }
    pub fn evaluate(&self, training_data: &Vec<Vec<Vec<f32>>>) -> f32 {
        let mut total_error: f32 = 0.0;
        let mut row: usize = 0;
        while { row < training_data.len() } {
            let predicted: Vec<f32> = self.predict(&training_data[row][0]);
            let mut i: usize = 0;
            while { i < self.output_size } {
                unsafe {
                    total_error = std::intrinsics::fadd_fast(
                        total_error,
                        std::intrinsics::fsub_fast(predicted[i], training_data[row][1][i]).abs(),
                    );
                }
                i += 1;
            }
            row += 1;
        }
        unsafe {
            std::intrinsics::fdiv_fast(total_error, (training_data.len() * self.output_size) as f32)
        }
    }
    pub fn optimise_current(
        &mut self,
        training_data: &Vec<Vec<Vec<f32>>>,
        rounds: u32,
        variations: u32,
    ) {
        let mut previous_score: f32 = self.evaluate(&training_data);
        for attempt in 0..rounds {
            let mut best_change: Vec<f32> = Vec::with_capacity(3);
            let mut new_score: f32 = previous_score + 1.0;
            let mut i: usize = 0;
            while { i < self.output_size } {
                let mut x: usize = 0;
                while { x < self.input_size } {
                    for _ in 0..variations {
                        let change: f32 = self.rng.gen_range(-0.0001, 0.0001);
                        self.nn[i][x] =
                            unsafe { std::intrinsics::fadd_fast(self.nn[i][x], change) };
                        let current_score: f32 = self.evaluate(&training_data);
                        if { current_score < new_score } {
                            new_score = current_score;
                            best_change = vec![i as f32, x as f32, change];
                        }
                        self.nn[i][x] =
                            unsafe { std::intrinsics::fsub_fast(self.nn[i][x], change) };
                    }
                    x += 1;
                }
                i += 1;
            }
            if { new_score < previous_score } {
                if { attempt % 500 == 0 } {
                    println!("Attempt {:?}, new score: {:?}", attempt, new_score);
                }

                self.nn[best_change[0] as usize][best_change[1] as usize] = unsafe {
                    std::intrinsics::fadd_fast(
                        self.nn[best_change[0] as usize][best_change[1] as usize],
                        best_change[2],
                    )
                };
                previous_score = new_score;
            } else {
                return;
            }
        }
    }
    pub fn train(&mut self, training_data: &Vec<Vec<Vec<f32>>>) {
        self.optimise_current(&training_data, 10, 10);
    }
}
fn main() {
    let mut the_machine: ML = ML::new(4, 2);
    println!("{:?}", the_machine.nn);
    println!("{:?}", the_machine.predict(&vec![1.0, 2.0, 3.0, 4.0]));
    let old_score: f32 =
        the_machine.evaluate(&vec![vec![vec![1.0, 2.0, 3.0, 4.0], vec![0.0, 1.0]]]);
    println!("{:?}", old_score);
    the_machine.optimise_current(
        &vec![vec![vec![1.0, 2.0, 3.0, 4.0], vec![0.0, 1.0]]],
        1000*1000,
        1000,
    );
    println!("{:?}", the_machine.nn);
    println!("{:?}", the_machine.predict(&vec![1.0, 2.0, 3.0, 4.0]));
    println!(
        "New score: {:?}, old score: {:?}",
        the_machine.evaluate(&vec![vec![vec![1.0, 2.0, 3.0, 4.0], vec![0.0, 1.0]]]),
        old_score
    );
}
