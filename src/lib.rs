#![feature(core_intrinsics)]
extern crate rand;
use rand::{thread_rng, Rng};

struct PredictionTempData {
    previous: Vec<f32>,
    next_layer: Vec<f32>,
}
impl PredictionTempData {
    pub fn new(largest_layer_capacity: usize) -> PredictionTempData {
        let previous: Vec<f32> = vec![0.0; largest_layer_capacity];
        let next_layer: Vec<f32> = vec![0.0; largest_layer_capacity];
        PredictionTempData {
            previous: previous,
            next_layer: next_layer,
        }
    }
}

pub struct ML {
    input_size: usize,
    output_size: usize,
    nn: Vec<f32>,
    sizes: Vec<[usize; 2]>,
    rng: rand::ThreadRng,
    total_evaluations: u64,
    largest_layer_capacity: usize,
}
impl ML {
    pub fn new(
        input_size: usize,
        output_size: usize,
        mut hidden_layers: usize,
        nodes_in_layer: usize,
    ) -> ML {
        assert!(input_size >= 1);
        assert!(output_size >= 1);
        assert!(hidden_layers >= 1);
        assert!(nodes_in_layer >= 1);
        hidden_layers -= 1; //to account for double counting
        let mut sizes: Vec<[usize; 2]> = vec![[input_size, nodes_in_layer]];
        for _ in 0..hidden_layers {
            sizes.push([nodes_in_layer, nodes_in_layer]);
        }
        sizes.push([nodes_in_layer, output_size]);
        let largest_layer_capacity: usize = if { nodes_in_layer > output_size } {
            nodes_in_layer
        } else {
            output_size
        };

        let mut new: ML = ML {
            input_size: input_size,
            output_size: output_size,
            nn: vec![
                0.0;
                input_size * nodes_in_layer
                    + hidden_layers * nodes_in_layer * nodes_in_layer
                    + nodes_in_layer * output_size
            ],
            sizes: sizes,
            rng: thread_rng(),
            total_evaluations: 0,
            largest_layer_capacity: largest_layer_capacity,
        };
        new.randomise();
        new
    }
    fn randomise(&mut self) {
        for i in 0..self.nn.len() {
            self.nn[i] = self.rng.gen_range(-1.0, 1.0);
        }
    }
    fn apply_layer(
        &self,
        input: &Vec<f32>,
        output: &mut Vec<f32>,
        size1: usize,
        size2: usize,
        offset: usize,
    ) {
        assert!(input.len() >= size1);
        assert!(output.len() >= size2);
        assert!(self.nn.len() >= offset + (size2 - 1) * size1 + (size1 - 1));
        for i in 0..size2 {
            let mut total: f32 = 0.0;
            let current_offset: usize = offset + i * size1;
            let mut x: usize = 0;
            for input_item in input.iter() {
                unsafe {
                    total = std::intrinsics::fadd_fast(
                        total,
                        std::intrinsics::fmul_fast(
                            *self.nn.get_unchecked(current_offset + x),
                            *input_item,
                        ),
                    );
                }
                x += 1;
            }
            unsafe {
                *output.get_unchecked_mut(i) =
                    std::intrinsics::fdiv_fast(total, std::intrinsics::fadd_fast(total.abs(), 1.0))
            };
        }
    }
    fn predict(&self, input: &Vec<f32>, temp_data: &mut PredictionTempData) {
        assert!(input.len() == self.input_size);
        let mut sizes_iter = self.sizes.iter();
        let sizes = sizes_iter.next().expect("");
        self.apply_layer(&input, &mut temp_data.previous, sizes[0], sizes[1], 0);
        let mut offset: usize = sizes[0] * sizes[1];
        for sizes in sizes_iter {
            self.apply_layer(
                &temp_data.previous,
                &mut temp_data.next_layer,
                sizes[0],
                sizes[1],
                offset,
            );
            std::mem::swap(&mut temp_data.previous, &mut temp_data.next_layer);
            offset += sizes[0] * sizes[1];
        }
    }
    pub fn predict_public(&self, input: &Vec<f32>) -> Vec<f32> {
        let mut temp_data: PredictionTempData =
            PredictionTempData::new(self.largest_layer_capacity);
        self.predict(&input, &mut temp_data);
        temp_data.previous[..self.output_size].to_vec()
    }
    fn evaluate(
        &mut self,
        training_data: &Vec<Vec<Vec<f32>>>,
        mut temp_data: &mut PredictionTempData,
    ) -> f32 {
        self.total_evaluations += 1;
        let mut total_error: f32 = 0.0;
        for row in 0..training_data.len() {
            let mut i: usize = 0;
            self.predict(&training_data[row][0], &mut temp_data);
            for item in temp_data.previous[..self.output_size].iter() {
                unsafe {
                    total_error = std::intrinsics::fadd_fast(
                        total_error,
                        std::intrinsics::fsub_fast(*item, training_data[row][1][i]).abs(),
                    );
                }
                i += 1;
            }
        }
        unsafe {
            std::intrinsics::fdiv_fast(total_error, (training_data.len() * self.output_size) as f32)
        }
    }
    pub fn optimise_current(&mut self, training_data: &Vec<Vec<Vec<f32>>>) {
        let mut temp_data: PredictionTempData =
            PredictionTempData::new(self.largest_layer_capacity);
        let mut previous_score: f32 = self.evaluate(&training_data, &mut temp_data);
        loop {
            let mut best_change_location: usize = 0;
            let mut best_change: f32 = 0.0;
            let mut new_score: f32 = previous_score;
            for location in 0..self.nn.len() {
                for options in [
                    [
                        1.0 / 16384.0,
                        1.0 / 8192.0,
                        1.0 / 4096.0,
                        1.0 / 2048.0,
                        1.0 / 1024.0,
                        1.0 / 512.0,
                        1.0 / 256.0,
                        1.0 / 128.0,
                        1.0 / 64.0,
                        1.0 / 32.0,
                        1.0 / 16.0,
                        1.0 / 8.0,
                        1.0 / 4.0,
                        1.0 / 2.0,
                        1.0,
                        2.0,
                        4.0,
                        8.0,
                        16.0,
                        32.0,
                    ],
                    [
                        -1.0 / 16384.0,
                        -1.0 / 8192.0,
                        -1.0 / 4096.0,
                        -1.0 / 2048.0,
                        -1.0 / 1024.0,
                        -1.0 / 512.0,
                        -1.0 / 256.0,
                        -1.0 / 128.0,
                        -1.0 / 64.0,
                        -1.0 / 32.0,
                        -1.0 / 16.0,
                        -1.0 / 8.0,
                        -1.0 / 4.0,
                        -1.0 / 2.0,
                        -1.0,
                        -2.0,
                        -4.0,
                        -8.0,
                        -16.0,
                        -32.0,
                    ],
                ]
                .iter()
                {
                    let mut last_current_score: f32 = previous_score;
                    for change in options.iter() {
                        let old: f32 = self.nn[location];
                        self.nn[location] += change;
                        let current_score: f32 = self.evaluate(&training_data, &mut temp_data);
                        self.nn[location] = old;
                        if { current_score < last_current_score } {
                            last_current_score = current_score;
                            if { current_score < new_score } {
                                new_score = current_score;
                                best_change = *change;
                                best_change_location = location;
                            }
                        } else {
                            break;
                        }
                    }
                }
            }
            if { new_score < previous_score } {
                self.nn[best_change_location] += best_change;
                previous_score = new_score;
            } else {
                return;
            }
        }
    }
    pub fn train(&mut self, training_data: &Vec<Vec<Vec<f32>>>, rounds: u32) {
        let mut temp_data: PredictionTempData =
            PredictionTempData::new(self.largest_layer_capacity);
        let mut best: Vec<f32> = self.nn.clone();
        let mut best_score: f32 = self.evaluate(&training_data, &mut temp_data);
        let mut total: f32 = 0.0;
        for round in 0..rounds {
            self.randomise();
            self.optimise_current(&training_data);
            let score: f32 = self.evaluate(&training_data, &mut temp_data);
            total += score;
            if { score < best_score } {
                best_score = score;
                best = self.nn.clone();
                println!(
                    "Round {:?}, new score: {:?}, total_evaluations: {:?} nn: {:?}",
                    round, score, self.total_evaluations, self.nn
                );
            } else {
                println!(
                    "Round {:?}, total_evaluations: {:?}, current best: {:?} avg: {:?}",
                    round,
                    self.total_evaluations,
                    best_score,
                    total / (round as f32 + 1.0)
                );
            }
        }
        self.nn = best;
    }
}
