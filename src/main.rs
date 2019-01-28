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
    pub fn randomise(&mut self) {
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
    pub fn predict(&self, input: Vec<f32>) -> Vec<f32> {
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
            }
        }
        return output;
    }
}
fn main() {
    let mut TheMachine: ML = ML::new(20, 7);
    println!("{:?}", TheMachine.nn);
}
