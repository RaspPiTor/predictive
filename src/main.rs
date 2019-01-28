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
                new_row.push(self.rng.gen());
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
            output.push(unsafe { std::intrinsics::fdiv_fast(fast_math::atan(total), 90.0) });
        }
        return output;
    }
    fn small_change(&mut self) {
        let mut i: usize = 0;
        while { i < self.output_size } {
            let mut x: usize = 0;
            while { x < self.input_size } {
                unsafe {
                    self.nn[i][x] = std::intrinsics::fadd_fast(
                        self.nn[i][x],
                        self.rng.gen_range(0.001, -0.001),
                    );
                }
                x += 1;
            }
            i += 1;
        }
    }
    pub fn evaluate(&self, training_data: Vec<Vec<Vec<f32>>>) {
        let mut total_error: f32 = 0.0;
        let mut row: usize = 0;
        while { row < training_data.len() } {
            let predicted: Vec<f32> = self.predict(&training_data[row][0]);
            let mut i: usize = 0;
            while { i < self.output_size } {
                unsafe {
                    total_error = std::intrinsics::fadd_fast(
                        total_error,
                        std::intrinsics::fdiv_fast(predicted[i], training_data[row][1][i]).abs(),
                    );
                        ;
                }
                i += 1;
            }
            row += 1;
        }
        total_error;
    }
    pub fn train(&mut self, training_data: Vec<Vec<Vec<f32>>>) {}
}
fn main() {
    let mut the_machine: ML = ML::new(4, 2);
    println!("{:?}", the_machine.nn);
    println!("{:?}", the_machine.predict(&vec![1.0, 2.0, 3.0, 4.0]));
}
